import os
import json
import time
import warnings

import numpy as np
from numpy import newaxis

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import EarlyStopping

configs = json.loads(open(os.path.join(os.path.dirname(__file__), 'configs.json')).read())
warnings.filterwarnings("ignore") #Hide messy Numpy warnings
seed = np.random.seed(seed=24)

def build_network(layers, data_gen_train, data_gen_test, steps_per_epoch, configs):
    model = Sequential()

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True)
    )
    model.add(Activation("tanh"))
    model.add(Dropout(
        0.5,
        seed=seed)
    )

    model.add(LSTM(
        layers[2],
        return_sequences=False)
    )
    model.add(Activation("tanh"))
    model.add(Dropout(
        0.5,
        seed=seed)
    )
    
    model.add(Dense(
        output_dim=layers[3])
    )
    model.add(Activation("linear"))


    start = time.time()
    model.compile(
        loss=configs['model']['loss_function'],
        optimizer=configs['model']['optimiser_function'],
        metrics=['mse', 'mae', 'mape', 'acc']
    )

    print("> Compilation Time : ", time.time() - start)

    start2 = time.time()

    history = model.fit_generator(
        data_gen_train,
        steps_per_epoch=steps_per_epoch,
        epochs=configs['model']['epochs'],
        callbacks=[EarlyStopping(monitor='mean_squared_error', min_delta=5e-5, patience=20, verbose=1)]
    )

    print("> Training Time : ", time.time() - start2)

    model.save(configs['model']['filename_model'])
    print('> Model Trained! Weights saved in', configs['model']['filename_model'])
    return model, history

def load_network(filename):
    #Load the h5 saved model and weights
    if(os.path.isfile(filename)):
        return load_model(filename)
    else:
        print('ERROR: "' + filename + '" file does not exist as a h5 model')
        return None
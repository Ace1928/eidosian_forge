import collections
import statistics
import numpy as np
from keras_tuner.src import backend
from keras_tuner.src import errors
from keras_tuner.src.backend import config
from keras_tuner.src.backend import keras
from keras_tuner.src.engine import hyperparameters as hp_module
from keras_tuner.src.engine import objective as obj_module
class TunerCallback(keras.callbacks.Callback):

    def __init__(self, tuner, trial):
        super().__init__()
        self.tuner = tuner
        self.trial = trial

    def on_epoch_begin(self, epoch, logs=None):
        self.tuner.on_epoch_begin(self.trial, self.model, epoch, logs=logs)

    def on_batch_begin(self, batch, logs=None):
        self.tuner.on_batch_begin(self.trial, self.model, batch, logs)

    def on_batch_end(self, batch, logs=None):
        self.tuner.on_batch_end(self.trial, self.model, batch, logs)

    def on_epoch_end(self, epoch, logs=None):
        self.tuner.on_epoch_end(self.trial, self.model, epoch, logs=logs)
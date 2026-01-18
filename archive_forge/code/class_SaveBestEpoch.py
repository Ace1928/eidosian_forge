import collections
import statistics
import numpy as np
from keras_tuner.src import backend
from keras_tuner.src import errors
from keras_tuner.src.backend import config
from keras_tuner.src.backend import keras
from keras_tuner.src.engine import hyperparameters as hp_module
from keras_tuner.src.engine import objective as obj_module
class SaveBestEpoch(keras.callbacks.Callback):
    """A Keras callback to save the model weights at the best epoch.

    Args:
        objective: An `Objective` instance.
        filepath: String. The file path to save the model weights.
    """

    def __init__(self, objective, filepath):
        super().__init__()
        self.objective = objective
        self.filepath = filepath
        if self.objective.direction == 'max':
            self.best_value = float('-inf')
        else:
            self.best_value = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        if not self.objective.has_value(logs):
            self._save_model()
            return
        current_value = self.objective.get_value(logs)
        if self.objective.better_than(current_value, self.best_value):
            self.best_value = current_value
            self._save_model()

    def _save_model(self):
        if config.backend() != 'tensorflow':
            self.model.save_weights(self.filepath)
            return
        write_filepath = backend.io.write_filepath(self.filepath, self.model.distribute_strategy)
        self.model.save_weights(write_filepath)
        backend.io.remove_temp_dir_with_filepath(write_filepath, self.model.distribute_strategy)
import collections
import copy
import os
import keras_tuner
from tensorflow import keras
from tensorflow import nest
from tensorflow.keras import callbacks as tf_callbacks
from tensorflow.keras.layers.experimental import preprocessing
from autokeras import pipeline as pipeline_module
from autokeras.utils import data_utils
from autokeras.utils import utils
def _build_best_model(self):
    best_trial = self.oracle.get_best_trials(1)[0]
    best_hp = best_trial.hyperparameters
    return self._try_build(best_hp)
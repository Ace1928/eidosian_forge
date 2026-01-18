import copy
import math
from keras_tuner.src import backend
from keras_tuner.src import utils
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import oracle as oracle_module
from keras_tuner.src.engine import tuner as tuner_module
def _compute_values_hash(self, values):
    values = copy.copy(values)
    values.pop('tuner/epochs', None)
    values.pop('tuner/initial_epoch', None)
    values.pop('tuner/bracket', None)
    values.pop('tuner/round', None)
    return super()._compute_values_hash(values)
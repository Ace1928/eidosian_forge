import copy
import math
from keras_tuner.src import backend
from keras_tuner.src import utils
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import oracle as oracle_module
from keras_tuner.src.engine import tuner as tuner_module
def _get_epochs(self, bracket_num, round_num):
    return math.ceil(self.max_epochs / self.factor ** (bracket_num - round_num))
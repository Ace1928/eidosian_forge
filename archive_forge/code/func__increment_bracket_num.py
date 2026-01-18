import copy
import math
from keras_tuner.src import backend
from keras_tuner.src import utils
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import oracle as oracle_module
from keras_tuner.src.engine import tuner as tuner_module
def _increment_bracket_num(self):
    self._current_bracket -= 1
    if self._current_bracket < 0:
        self._current_bracket = self._get_num_brackets() - 1
        self._current_iteration += 1
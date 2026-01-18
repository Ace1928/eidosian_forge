import copy
import math
from keras_tuner.src import backend
from keras_tuner.src import utils
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import oracle as oracle_module
from keras_tuner.src.engine import tuner as tuner_module
def _start_new_bracket(self):
    rounds = []
    rounds.extend(([] for _ in range(self._get_num_rounds(self._current_bracket))))
    bracket = {'bracket_num': self._current_bracket, 'rounds': rounds}
    self._brackets.append(bracket)
import re
import warnings
import numpy as np
from keras.src import backend
from keras.src import initializers
from keras.src import ops
from keras.src.optimizers.schedules import learning_rate_schedule
from keras.src.saving import serialization_lib
from keras.src.utils import tracking
from keras.src.utils.naming import auto_name
def _get_current_learning_rate(self):
    if isinstance(self._learning_rate, learning_rate_schedule.LearningRateSchedule):
        return self._learning_rate(self.iterations)
    elif callable(self._learning_rate):
        return self._learning_rate(self.iterations)
    return self._learning_rate
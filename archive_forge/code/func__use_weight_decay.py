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
def _use_weight_decay(self, variable):
    exclude_from_weight_decay = getattr(self, '_exclude_from_weight_decay', [])
    exclude_from_weight_decay_names = getattr(self, '_exclude_from_weight_decay_names', [])
    variable_id = self._var_key(variable)
    for exclude_id in exclude_from_weight_decay:
        if variable_id == exclude_id:
            return False
    for name in exclude_from_weight_decay_names:
        if re.search(name, variable.name) is not None:
            return False
    return True
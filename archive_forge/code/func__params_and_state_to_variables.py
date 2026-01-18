import inspect
import numpy as np
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.saving import serialization_lib
from keras.src.utils import jax_utils
from keras.src.utils import tracking
from keras.src.utils import tree
from keras.src.utils.module_utils import jax
def _params_and_state_to_variables(self, params, state):
    if params:
        if state:
            return {**params, **state}
        else:
            return params
    elif state:
        return state
    return {}
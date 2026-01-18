import contextlib
import os
import ml_dtypes
import numpy as np
import torch
import tree
from keras.src.backend.common import KerasVariable
from keras.src.backend.common import global_state
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.dtypes import result_type
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.common.stateless_scope import StatelessScope
from keras.src.backend.config import floatx
from keras.src.utils.nest import pack_sequence_as
@classmethod
def __torch_function__(cls, func, types, args=(), kwargs=None):
    args = [arg.value if isinstance(arg, KerasVariable) else arg for arg in args]
    if kwargs is None:
        kwargs = {}
    kwargs = {key: value.value if isinstance(value, KerasVariable) else value for key, value in kwargs.items()}
    return func(*args, **kwargs)
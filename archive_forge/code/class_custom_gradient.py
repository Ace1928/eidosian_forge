import contextlib
import os
import ml_dtypes
import numpy as np
import torch
from keras.src.backend.common import KerasVariable
from keras.src.backend.common import global_state
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.dtypes import result_type
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.common.stateless_scope import StatelessScope
from keras.src.backend.config import floatx
from keras.src.utils import tree
class custom_gradient:
    """Decorator for custom gradients.

    Args:
        forward_fn: Forward pass function.
    """

    def __init__(self, forward_fn):
        self.forward_fn = forward_fn

    def __call__(self, *args, **kwargs):
        return CustomGradientFunction.apply(self.forward_fn, *args, **kwargs)
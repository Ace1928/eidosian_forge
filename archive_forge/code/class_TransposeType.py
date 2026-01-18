import os
import re
import numpy
from .utils import ExplicitEnum, expand_dims, is_numpy_array, is_torch_tensor, logging, reshape, squeeze, tensor_size
from .utils import transpose as transpose_func
class TransposeType(ExplicitEnum):
    """
    Possible ...
    """
    NO = 'no'
    SIMPLE = 'simple'
    CONV1D = 'conv1d'
    CONV2D = 'conv2d'
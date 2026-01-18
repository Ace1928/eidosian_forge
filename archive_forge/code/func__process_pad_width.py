import builtins
import re
import numpy as np
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.backend.common import dtypes
from keras.src.ops import operation_utils
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import broadcast_shapes
from keras.src.ops.operation_utils import reduce_shape
def _process_pad_width(self, pad_width):
    if isinstance(pad_width, int):
        return ((pad_width, pad_width),)
    if isinstance(pad_width, (tuple, list)) and isinstance(pad_width[0], int):
        return (pad_width,)
    first_len = len(pad_width[0])
    for i, pw in enumerate(pad_width):
        if len(pw) != first_len:
            raise ValueError(f'`pad_width` should be a list of tuples of length 1 or 2. Received: pad_width={pad_width}')
        if len(pw) == 1:
            pad_width[i] = (pw[0], pw[0])
    return pad_width
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
class Pad(Operation):

    def __init__(self, pad_width, mode='constant'):
        super().__init__()
        self.pad_width = self._process_pad_width(pad_width)
        self.mode = mode

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

    def call(self, x, constant_values=None):
        return backend.numpy.pad(x, pad_width=self.pad_width, mode=self.mode, constant_values=constant_values)

    def compute_output_spec(self, x, constant_values=None):
        output_shape = list(x.shape)
        if len(self.pad_width) == 1:
            pad_width = [self.pad_width[0] for _ in range(len(output_shape))]
        elif len(self.pad_width) == len(output_shape):
            pad_width = self.pad_width
        else:
            raise ValueError(f'`pad_width` must have the same length as `x.shape`. Received: pad_width={self.pad_width} (of length {len(self.pad_width)}) and x.shape={x.shape} (of length {len(x.shape)})')
        for i in range(len(output_shape)):
            if output_shape[i] is None:
                output_shape[i] = None
            else:
                output_shape[i] += pad_width[i][0] + pad_width[i][1]
        return KerasTensor(output_shape, dtype=x.dtype)
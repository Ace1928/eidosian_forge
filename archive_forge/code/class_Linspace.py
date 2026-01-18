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
class Linspace(Operation):

    def __init__(self, num=50, endpoint=True, retstep=False, dtype=float, axis=0):
        super().__init__()
        self.num = num
        self.endpoint = endpoint
        self.retstep = retstep
        self.dtype = dtype
        self.axis = axis

    def call(self, start, stop):
        return backend.numpy.linspace(start, stop, num=self.num, endpoint=self.endpoint, retstep=self.retstep, dtype=self.dtype, axis=self.axis)

    def compute_output_spec(self, start, stop):
        start_shape = getattr(start, 'shape', [])
        stop_shape = getattr(stop, 'shape', [])
        output_shape = broadcast_shapes(start_shape, stop_shape)
        if self.axis == -1:
            output_shape = output_shape + [self.num]
        elif self.axis >= 0:
            output_shape = output_shape[:self.axis] + [self.num] + output_shape[self.axis:]
        else:
            output_shape = output_shape[:self.axis + 1] + [self.num] + output_shape[self.axis + 1:]
        dtype = self.dtype if self.dtype is not None else getattr(start, 'dtype', type(start))
        dtype = backend.result_type(dtype, float)
        if self.retstep:
            return (KerasTensor(output_shape, dtype=dtype), None)
        return KerasTensor(output_shape, dtype=dtype)
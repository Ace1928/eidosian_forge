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
class CountNonzero(Operation):

    def __init__(self, axis=None):
        super().__init__()
        if isinstance(axis, int):
            self.axis = (axis,)
        else:
            self.axis = axis

    def call(self, x):
        return backend.numpy.count_nonzero(x, axis=self.axis)

    def compute_output_spec(self, x):
        return KerasTensor(reduce_shape(x.shape, axis=self.axis), dtype='int32')
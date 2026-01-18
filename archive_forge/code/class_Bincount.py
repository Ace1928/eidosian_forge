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
class Bincount(Operation):

    def __init__(self, weights=None, minlength=0):
        super().__init__()
        self.weights = weights
        self.minlength = minlength

    def call(self, x):
        return backend.numpy.bincount(x, weights=self.weights, minlength=self.minlength)

    def compute_output_spec(self, x):
        dtypes_to_resolve = [x.dtype]
        if self.weights is not None:
            weights = backend.convert_to_tensor(self.weights)
            dtypes_to_resolve.append(weights.dtype)
            dtype = dtypes.result_type(*dtypes_to_resolve)
        else:
            dtype = 'int32'
        return KerasTensor(list(x.shape[:-1]) + [None], dtype=dtype)
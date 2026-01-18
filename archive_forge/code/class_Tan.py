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
class Tan(Operation):

    def call(self, x):
        return backend.numpy.tan(x)

    def compute_output_spec(self, x):
        dtype = backend.standardize_dtype(getattr(x, 'dtype', backend.floatx()))
        if dtype == 'int64':
            dtype = backend.floatx()
        else:
            dtype = dtypes.result_type(dtype, float)
        sparse = getattr(x, 'sparse', False)
        return KerasTensor(x.shape, dtype=dtype, sparse=sparse)
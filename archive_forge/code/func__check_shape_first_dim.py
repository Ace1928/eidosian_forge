from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.backend import standardize_data_format
from keras.src.backend.common.backend_utils import (
from keras.src.ops import operation_utils
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import reduce_shape
def _check_shape_first_dim(self, name1, shape1, name2, shape2):
    if shape1[0] != shape2[0]:
        raise ValueError(f'Arguments `{name1}` and `{name2}` must have the same first dimension. Received shapes: `{shape1}` and `{shape2}`.')
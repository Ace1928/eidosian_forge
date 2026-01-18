from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import reduce_shape
def _inv(x):
    x = backend.convert_to_tensor(x)
    _assert_2d(x)
    _assert_square(x)
    return backend.linalg.inv(x)
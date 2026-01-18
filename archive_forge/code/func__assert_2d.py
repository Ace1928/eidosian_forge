from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import reduce_shape
def _assert_2d(*arrays):
    for a in arrays:
        if a.ndim < 2:
            raise ValueError('Expected input to have rank >= 2. Received input with shape {a.shape}.')
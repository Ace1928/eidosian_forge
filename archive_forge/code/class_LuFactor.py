from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import reduce_shape
class LuFactor(Operation):

    def __init__(self):
        super().__init__()

    def call(self, x):
        return _lu_factor(x)

    def compute_output_spec(self, x):
        _assert_2d(x)
        batch_shape = x.shape[:-2]
        m, n = x.shape[-2:]
        k = min(m, n)
        return (KerasTensor(batch_shape + (m, n), x.dtype), KerasTensor(batch_shape + (k,), x.dtype))
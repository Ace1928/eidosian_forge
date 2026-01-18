from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import reduce_shape
class RFFT(Operation):

    def __init__(self, fft_length=None):
        super().__init__()
        self.fft_length = fft_length

    def compute_output_spec(self, x):
        if len(x.shape) < 1:
            raise ValueError(f'Input should have rank >= 1. Received: input.shape = {x.shape}')
        if self.fft_length is not None:
            new_last_dimension = self.fft_length // 2 + 1
        elif x.shape[-1] is not None:
            new_last_dimension = x.shape[-1] // 2 + 1
        else:
            new_last_dimension = None
        new_shape = x.shape[:-1] + (new_last_dimension,)
        return (KerasTensor(shape=new_shape, dtype=x.dtype), KerasTensor(shape=new_shape, dtype=x.dtype))

    def call(self, x):
        return backend.math.rfft(x, fft_length=self.fft_length)
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import reduce_shape
class STFT(Operation):

    def __init__(self, sequence_length, sequence_stride, fft_length, window='hann', center=True):
        super().__init__()
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.fft_length = fft_length
        self.window = window
        self.center = center

    def compute_output_spec(self, x):
        if x.shape[-1] is not None:
            padded = 0 if self.center is False else self.fft_length // 2 * 2
            num_sequences = 1 + (x.shape[-1] + padded - self.fft_length) // self.sequence_stride
        else:
            num_sequences = None
        new_shape = x.shape[:-1] + (num_sequences, self.fft_length // 2 + 1)
        return (KerasTensor(shape=new_shape, dtype=x.dtype), KerasTensor(shape=new_shape, dtype=x.dtype))

    def call(self, x):
        return backend.math.stft(x, sequence_length=self.sequence_length, sequence_stride=self.sequence_stride, fft_length=self.fft_length, window=self.window, center=self.center)
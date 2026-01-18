from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import reduce_shape
class ExtractSequences(Operation):

    def __init__(self, sequence_length, sequence_stride):
        super().__init__()
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride

    def compute_output_spec(self, x):
        if len(x.shape) < 1:
            raise ValueError(f'Input should have rank >= 1. Received: input.shape = {x.shape}')
        if x.shape[-1] is not None:
            num_sequences = 1 + (x.shape[-1] - self.sequence_length) // self.sequence_stride
        else:
            num_sequences = None
        new_shape = x.shape[:-1] + (num_sequences, self.sequence_length)
        return KerasTensor(shape=new_shape, dtype=x.dtype)

    def call(self, x):
        return backend.math.extract_sequences(x, sequence_length=self.sequence_length, sequence_stride=self.sequence_stride)
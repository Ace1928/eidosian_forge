import numpy as np
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.utils import traceback_utils
class Slice(Operation):

    def call(self, inputs, start_indices, shape):
        return backend.core.slice(inputs, start_indices, shape)

    def compute_output_spec(self, inputs, start_indices, shape):
        return KerasTensor(shape, dtype=inputs.dtype)
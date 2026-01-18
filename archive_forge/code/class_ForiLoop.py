import numpy as np
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.utils import traceback_utils
class ForiLoop(Operation):

    def __init__(self, lower, upper, body_fun):
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.body_fun = body_fun

    def call(self, init_val):
        return backend.core.fori_loop(self.lower, self.upper, self.body_fun, init_val)

    def compute_output_spec(self, init_val):
        return KerasTensor(init_val.shape, dtype=init_val.dtype)
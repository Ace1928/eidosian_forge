import numpy as np
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.utils import traceback_utils
class WhileLoop(Operation):

    def __init__(self, cond, body, maximum_iterations):
        super().__init__()
        self.cond = cond
        self.body = body
        self.maximum_iterations = maximum_iterations

    def call(self, loop_vars):
        return backend.core.while_loop(self.cond, self.body, loop_vars, maximum_iterations=self.maximum_iterations)

    def compute_output_spec(self, loop_vars):
        return [KerasTensor(v.shape, dtype=v.dtype) for v in loop_vars]
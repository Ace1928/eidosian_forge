from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import reduce_shape
class FFT2(Operation):

    def __init__(self):
        super().__init__()
        self.axes = (-2, -1)

    def compute_output_spec(self, x):
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            raise ValueError(f'Input `x` should be a tuple of two tensors - real and imaginary. Received: x={x}')
        real, imag = x
        if real.shape != imag.shape:
            raise ValueError(f'Input `x` should be a tuple of two tensors - real and imaginary. Both the real and imaginary parts should have the same shape. Received: x[0].shape = {real.shape}, x[1].shape = {imag.shape}')
        if len(real.shape) < 2:
            raise ValueError(f'Input should have rank >= 2. Received: input.shape = {real.shape}')
        m = real.shape[self.axes[0]]
        n = real.shape[self.axes[1]]
        if m is None or n is None:
            raise ValueError(f'Input should have its {self.axes} axes fully-defined. Received: input.shape = {real.shape}')
        return (KerasTensor(shape=real.shape, dtype=real.dtype), KerasTensor(shape=imag.shape, dtype=imag.dtype))

    def call(self, x):
        return backend.math.fft2(x)
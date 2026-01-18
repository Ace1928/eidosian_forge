import builtins
import re
import numpy as np
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.backend.common import dtypes
from keras.src.ops import operation_utils
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import broadcast_shapes
from keras.src.ops.operation_utils import reduce_shape
class Split(Operation):

    def __init__(self, indices_or_sections, axis=0):
        super().__init__()
        if not isinstance(indices_or_sections, int):
            indices_or_sections = tuple(indices_or_sections)
        self.indices_or_sections = indices_or_sections
        self.axis = axis

    def call(self, x):
        return backend.numpy.split(x, self.indices_or_sections, axis=self.axis)

    def compute_output_spec(self, x):
        x_shape = list(x.shape)
        x_size_on_axis = x_shape[self.axis]
        if isinstance(self.indices_or_sections, int):
            if x_size_on_axis is None:
                x_shape[self.axis] = None
                return [KerasTensor(x_shape, dtype=x.dtype) for _ in range(self.indices_or_sections)]
            if np.mod(x_size_on_axis, self.indices_or_sections) != 0:
                raise ValueError(f'`x` size on given `axis` must be dividible by `indices_or_sections` when `indices_or_sections` is an int. But received {x_size_on_axis} and {self.indices_or_sections}.')
            size = x_size_on_axis // self.indices_or_sections
            x_shape[self.axis] = size
            return [KerasTensor(x_shape, dtype=x.dtype) for _ in range(self.indices_or_sections)]
        indices_or_sections = (0, *self.indices_or_sections, x_size_on_axis)
        output_size = np.diff(indices_or_sections)
        outputs = []
        for i in range(len(output_size)):
            output_shape = list(x_shape)
            output_shape[self.axis] = int(output_size[i])
            outputs.append(KerasTensor(output_shape, dtype=x.dtype))
        return outputs
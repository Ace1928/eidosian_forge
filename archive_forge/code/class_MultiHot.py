from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.backend import standardize_data_format
from keras.src.backend.common.backend_utils import (
from keras.src.ops import operation_utils
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import reduce_shape
class MultiHot(Operation):

    def __init__(self, num_classes=None, axis=-1, dtype=None, name=None, **kwargs):
        if num_classes is None and 'num_tokens' in kwargs:
            num_classes = kwargs.pop('num_tokens')
        if num_classes is None:
            raise ValueError('Argument `num_classes` must be specified.')
        super().__init__(name, **kwargs)
        self.num_classes = num_classes
        self.axis = axis
        self.dtype = dtype or backend.floatx()

    def call(self, inputs):
        return backend.nn.multi_hot(inputs, num_classes=self.num_classes, axis=self.axis, dtype=self.dtype)

    def compute_output_spec(self, inputs):
        x_shape = list(getattr(inputs, 'shape', []))
        if self.axis == -1:
            x_shape.append(self.num_classes)
        elif self.axis >= 0 and self.axis < len(x_shape):
            x_shape.insert(self.axis, self.num_classes)
        else:
            raise ValueError(f'axis must be -1 or between [0, {len(inputs.shape)}), but received {self.axis}.')
        if len(x_shape) == 2:
            x_shape = [x_shape[-1]]
        else:
            x_shape = [x_shape[0]] + x_shape[2:]
        return KerasTensor(x_shape, dtype=inputs.dtype)
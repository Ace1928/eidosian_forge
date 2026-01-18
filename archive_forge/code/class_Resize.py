from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import compute_conv_output_shape
class Resize(Operation):

    def __init__(self, size, interpolation='bilinear', antialias=False, data_format='channels_last'):
        super().__init__()
        self.size = tuple(size)
        self.interpolation = interpolation
        self.antialias = antialias
        self.data_format = data_format

    def call(self, image):
        return backend.image.resize(image, self.size, interpolation=self.interpolation, antialias=self.antialias, data_format=self.data_format)

    def compute_output_spec(self, image):
        if len(image.shape) == 3:
            return KerasTensor(self.size + (image.shape[-1],), dtype=image.dtype)
        elif len(image.shape) == 4:
            if self.data_format == 'channels_last':
                return KerasTensor((image.shape[0],) + self.size + (image.shape[-1],), dtype=image.dtype)
            else:
                return KerasTensor((image.shape[0], image.shape[1]) + self.size, dtype=image.dtype)
        raise ValueError(f'Invalid input rank: expected rank 3 (single image) or rank 4 (batch of images). Received input with shape: image.shape={image.shape}')
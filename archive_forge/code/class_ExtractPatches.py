from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import compute_conv_output_shape
class ExtractPatches(Operation):

    def __init__(self, size, strides=None, dilation_rate=1, padding='valid', data_format='channels_last'):
        super().__init__()
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.padding = padding
        self.data_format = data_format

    def call(self, image):
        return _extract_patches(image=image, size=self.size, strides=self.strides, dilation_rate=self.dilation_rate, padding=self.padding, data_format=self.data_format)

    def compute_output_spec(self, image):
        image_shape = image.shape
        if not self.strides:
            strides = (self.size[0], self.size[1])
        if self.data_format == 'channels_last':
            channels_in = image.shape[-1]
        else:
            channels_in = image.shape[-3]
        if len(image.shape) == 3:
            image_shape = (1,) + image_shape
        filters = self.size[0] * self.size[1] * channels_in
        kernel_size = (self.size[0], self.size[1])
        out_shape = compute_conv_output_shape(image_shape, filters, kernel_size, strides=strides, padding=self.padding, data_format=self.data_format, dilation_rate=self.dilation_rate)
        if len(image.shape) == 3:
            out_shape = out_shape[1:]
        return KerasTensor(shape=out_shape, dtype=image.dtype)
import tensorflow.compat.v2 as tf
from keras.src.engine.base_layer import Layer
from keras.src.engine.input_spec import InputSpec
from keras.src.utils import conv_utils
def _validate_reduction_axis(self, input_shape, axes):
    for axis in axes:
        if input_shape[axis] == 0:
            raise ValueError(f'Incorrect input shape {input_shape} with dimension 0 at reduction axis {axis}.')
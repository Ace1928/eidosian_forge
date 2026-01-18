import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.util.tf_export import keras_export
from keras.src import backend
from keras.src.engine import base_layer
from keras.src.engine import base_preprocessing_layer
from keras.src.layers.preprocessing import preprocessing_utils as utils
from keras.src.utils import image_utils
from keras.src.utils import tf_utils
@keras_export('keras.layers.RandomZoom', 'keras.layers.experimental.preprocessing.RandomZoom', v1=[])
class RandomZoom(base_layer.BaseRandomLayer):
    """A preprocessing layer which randomly zooms images during training.

    This layer will randomly zoom in or out on each axis of an image
    independently, filling empty space according to `fill_mode`.

    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    of integer or floating point dtype.
    By default, the layer will output floats.

    For an overview and full list of preprocessing layers, see the preprocessing
    [guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

    Args:
        height_factor: a float represented as fraction of value,
            or a tuple of size 2 representing lower and upper bound
            for zooming vertically. When represented as a single float,
            this value is used for both the upper and
            lower bound. A positive value means zooming out,
            while a negative value
            means zooming in. For instance, `height_factor=(0.2, 0.3)`
            result in an output zoomed out by a random amount
            in the range `[+20%, +30%]`.
            `height_factor=(-0.3, -0.2)` result in an output zoomed
            in by a random amount in the range `[+20%, +30%]`.
        width_factor: a float represented as fraction of value,
            or a tuple of size 2 representing lower and upper bound
            for zooming horizontally. When
            represented as a single float, this value is used
            for both the upper and
            lower bound. For instance, `width_factor=(0.2, 0.3)`
            result in an output
            zooming out between 20% to 30%.
            `width_factor=(-0.3, -0.2)` result in an
            output zooming in between 20% to 30%. `None` means
            i.e., zooming vertical and horizontal directions
            by preserving the aspect ratio. Defaults to `None`.
        fill_mode: Points outside the boundaries of the input are
            filled according to the given mode
            (one of `{"constant", "reflect", "wrap", "nearest"}`).
            - *reflect*: `(d c b a | a b c d | d c b a)`
                The input is extended by reflecting about
                the edge of the last pixel.
            - *constant*: `(k k k k | a b c d | k k k k)`
                The input is extended by filling all values beyond
                the edge with the same constant value k = 0.
            - *wrap*: `(a b c d | a b c d | a b c d)` The input is extended by
                wrapping around to the opposite edge.
            - *nearest*: `(a a a a | a b c d | d d d d)`
                The input is extended by the nearest pixel.
        interpolation: Interpolation mode. Supported values: `"nearest"`,
            `"bilinear"`.
        seed: Integer. Used to create a random seed.
        fill_value: a float represents the value to be filled outside
            the boundaries when `fill_mode="constant"`.

    Example:

    >>> input_img = np.random.random((32, 224, 224, 3))
    >>> layer = tf.keras.layers.RandomZoom(.5, .2)
    >>> out_img = layer(input_img)
    >>> out_img.shape
    TensorShape([32, 224, 224, 3])

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format.

    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format.
    """

    def __init__(self, height_factor, width_factor=None, fill_mode='reflect', interpolation='bilinear', seed=None, fill_value=0.0, **kwargs):
        base_preprocessing_layer.keras_kpl_gauge.get_cell('RandomZoom').set(True)
        super().__init__(seed=seed, force_generator=True, **kwargs)
        self.height_factor = height_factor
        if isinstance(height_factor, (tuple, list)):
            self.height_lower = height_factor[0]
            self.height_upper = height_factor[1]
        else:
            self.height_lower = -height_factor
            self.height_upper = height_factor
        if abs(self.height_lower) > 1.0 or abs(self.height_upper) > 1.0:
            raise ValueError(f'`height_factor` argument must have values between [-1, 1]. Received: height_factor={height_factor}')
        self.width_factor = width_factor
        if width_factor is not None:
            if isinstance(width_factor, (tuple, list)):
                self.width_lower = width_factor[0]
                self.width_upper = width_factor[1]
            else:
                self.width_lower = -width_factor
                self.width_upper = width_factor
            if self.width_lower < -1.0 or self.width_upper < -1.0:
                raise ValueError(f'`width_factor` argument must have values larger than -1. Received: width_factor={width_factor}')
        check_fill_mode_and_interpolation(fill_mode, interpolation)
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.interpolation = interpolation
        self.seed = seed

    def call(self, inputs, training=True):
        inputs = convert_inputs(inputs, self.compute_dtype)

        def random_zoomed_inputs(inputs):
            """Zoomed inputs with random ops."""
            original_shape = inputs.shape
            unbatched = inputs.shape.rank == 3
            if unbatched:
                inputs = tf.expand_dims(inputs, 0)
            inputs_shape = tf.shape(inputs)
            batch_size = inputs_shape[0]
            img_hd = tf.cast(inputs_shape[H_AXIS], tf.float32)
            img_wd = tf.cast(inputs_shape[W_AXIS], tf.float32)
            height_zoom = self._random_generator.random_uniform(shape=[batch_size, 1], minval=1.0 + self.height_lower, maxval=1.0 + self.height_upper)
            if self.width_factor is not None:
                width_zoom = self._random_generator.random_uniform(shape=[batch_size, 1], minval=1.0 + self.width_lower, maxval=1.0 + self.width_upper)
            else:
                width_zoom = height_zoom
            zooms = tf.cast(tf.concat([width_zoom, height_zoom], axis=1), dtype=tf.float32)
            output = transform(inputs, get_zoom_matrix(zooms, img_hd, img_wd), fill_mode=self.fill_mode, fill_value=self.fill_value, interpolation=self.interpolation)
            if unbatched:
                output = tf.squeeze(output, 0)
            output.set_shape(original_shape)
            return output
        if training:
            return random_zoomed_inputs(inputs)
        else:
            return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'height_factor': self.height_factor, 'width_factor': self.width_factor, 'fill_mode': self.fill_mode, 'fill_value': self.fill_value, 'interpolation': self.interpolation, 'seed': self.seed}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
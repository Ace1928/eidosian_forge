from typing import Optional
import numpy as np
from ray.rllib.algorithms.dreamerv3.utils import (
from ray.rllib.utils.framework import try_import_tf
class ConvTransposeAtari(tf.keras.Model):
    """A Conv2DTranspose decoder to generate Atari images from a latent space.

    Wraps an initial single linear layer with a stack of 4 Conv2DTranspose layers (with
    layer normalization) and a diag Gaussian, from which we then sample the final image.
    Sampling is done with a fixed stddev=1.0 and using the mean values coming from the
    last Conv2DTranspose layer.
    """

    def __init__(self, *, model_size: Optional[str]='XS', cnn_multiplier: Optional[int]=None, gray_scaled: bool):
        """Initializes a ConvTransposeAtari instance.

        Args:
            model_size: The "Model Size" used according to [1] Appendinx B.
                Use None for manually setting the `cnn_multiplier`.
            cnn_multiplier: Optional override for the additional factor used to multiply
                the number of filters with each CNN transpose layer. Starting with
                8 * `cnn_multiplier` filters in the first CNN transpose layer, the
                number of filters then decreases via `4*cnn_multiplier`,
                `2*cnn_multiplier`, till `1*cnn_multiplier`.
            gray_scaled: Whether the last Conv2DTranspose layer's output has only 1
                color channel (gray_scaled=True) or 3 RGB channels (gray_scaled=False).
        """
        super().__init__(name='image_decoder')
        self.model_size = model_size
        cnn_multiplier = get_cnn_multiplier(self.model_size, override=cnn_multiplier)
        self.input_dims = (4, 4, 8 * cnn_multiplier)
        self.gray_scaled = gray_scaled
        self.dense_layer = tf.keras.layers.Dense(units=int(np.prod(self.input_dims)), activation=None, use_bias=True)
        self.conv_transpose_layers = [tf.keras.layers.Conv2DTranspose(filters=4 * cnn_multiplier, kernel_size=4, strides=(2, 2), padding='same', activation=None, use_bias=False), tf.keras.layers.Conv2DTranspose(filters=2 * cnn_multiplier, kernel_size=4, strides=(2, 2), padding='same', activation=None, use_bias=False), tf.keras.layers.Conv2DTranspose(filters=1 * cnn_multiplier, kernel_size=4, strides=(2, 2), padding='same', activation=None, use_bias=False)]
        self.layer_normalizations = []
        for _ in range(len(self.conv_transpose_layers)):
            self.layer_normalizations.append(tf.keras.layers.LayerNormalization())
        self.output_conv2d_transpose = tf.keras.layers.Conv2DTranspose(filters=1 if self.gray_scaled else 3, kernel_size=4, strides=(2, 2), padding='same', activation=None, use_bias=True)
        dl_type = tf.keras.mixed_precision.global_policy().compute_dtype or tf.float32
        self.call = tf.function(input_signature=[tf.TensorSpec(shape=[None, get_gru_units(model_size)], dtype=dl_type), tf.TensorSpec(shape=[None, get_num_z_categoricals(model_size), get_num_z_classes(model_size)], dtype=dl_type)])(self.call)

    def call(self, h, z):
        """Performs a forward pass through the Conv2D transpose decoder.

        Args:
            h: The deterministic hidden state of the sequence model.
            z: The sequence of stochastic discrete representations of the original
                observation input. Note: `z` is not used for the dynamics predictor
                model (which predicts z from h).
        """
        assert len(z.shape) == 3
        z_shape = tf.shape(z)
        z = tf.reshape(z, shape=(z_shape[0], -1))
        assert len(z.shape) == 2
        input_ = tf.concat([h, z], axis=-1)
        input_.set_shape([None, get_num_z_categoricals(self.model_size) * get_num_z_classes(self.model_size) + get_gru_units(self.model_size)])
        out = self.dense_layer(input_)
        out = tf.reshape(out, shape=(-1,) + self.input_dims)
        for conv_transpose_2d, layer_norm in zip(self.conv_transpose_layers, self.layer_normalizations):
            out = tf.nn.silu(layer_norm(inputs=conv_transpose_2d(out)))
        out = self.output_conv2d_transpose(out)
        out += 0.5
        out_shape = tf.shape(out)
        loc = tf.reshape(out, shape=(out_shape[0], -1))
        return loc
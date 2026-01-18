from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list
from ...utils import logging
from .configuration_regnet import RegNetConfig
class TFRegNetEmbeddings(keras.layers.Layer):
    """
    RegNet Embeddings (stem) composed of a single aggressive convolution.
    """

    def __init__(self, config: RegNetConfig, **kwargs):
        super().__init__(**kwargs)
        self.num_channels = config.num_channels
        self.embedder = TFRegNetConvLayer(in_channels=config.num_channels, out_channels=config.embedding_size, kernel_size=3, stride=2, activation=config.hidden_act, name='embedder')

    def call(self, pixel_values):
        num_channels = shape_list(pixel_values)[1]
        if tf.executing_eagerly() and num_channels != self.num_channels:
            raise ValueError('Make sure that the channel dimension of the pixel values match with the one set in the configuration.')
        pixel_values = tf.transpose(pixel_values, perm=(0, 2, 3, 1))
        hidden_state = self.embedder(pixel_values)
        return hidden_state

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'embedder', None) is not None:
            with tf.name_scope(self.embedder.name):
                self.embedder.build(None)
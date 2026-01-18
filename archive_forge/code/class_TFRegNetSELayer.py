from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list
from ...utils import logging
from .configuration_regnet import RegNetConfig
class TFRegNetSELayer(keras.layers.Layer):
    """
    Squeeze and Excitation layer (SE) proposed in [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507).
    """

    def __init__(self, in_channels: int, reduced_channels: int, **kwargs):
        super().__init__(**kwargs)
        self.pooler = keras.layers.GlobalAveragePooling2D(keepdims=True, name='pooler')
        self.attention = [keras.layers.Conv2D(filters=reduced_channels, kernel_size=1, activation='relu', name='attention.0'), keras.layers.Conv2D(filters=in_channels, kernel_size=1, activation='sigmoid', name='attention.2')]
        self.in_channels = in_channels
        self.reduced_channels = reduced_channels

    def call(self, hidden_state):
        pooled = self.pooler(hidden_state)
        for layer_module in self.attention:
            pooled = layer_module(pooled)
        hidden_state = hidden_state * pooled
        return hidden_state

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'pooler', None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build((None, None, None, None))
        if getattr(self, 'attention', None) is not None:
            with tf.name_scope(self.attention[0].name):
                self.attention[0].build([None, None, None, self.in_channels])
            with tf.name_scope(self.attention[1].name):
                self.attention[1].build([None, None, None, self.reduced_channels])
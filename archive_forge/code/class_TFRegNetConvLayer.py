from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list
from ...utils import logging
from .configuration_regnet import RegNetConfig
class TFRegNetConvLayer(keras.layers.Layer):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3, stride: int=1, groups: int=1, activation: Optional[str]='relu', **kwargs):
        super().__init__(**kwargs)
        self.padding = keras.layers.ZeroPadding2D(padding=kernel_size // 2)
        self.convolution = keras.layers.Conv2D(filters=out_channels, kernel_size=kernel_size, strides=stride, padding='VALID', groups=groups, use_bias=False, name='convolution')
        self.normalization = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.9, name='normalization')
        self.activation = ACT2FN[activation] if activation is not None else tf.identity
        self.in_channels = in_channels
        self.out_channels = out_channels

    def call(self, hidden_state):
        hidden_state = self.convolution(self.padding(hidden_state))
        hidden_state = self.normalization(hidden_state)
        hidden_state = self.activation(hidden_state)
        return hidden_state

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'convolution', None) is not None:
            with tf.name_scope(self.convolution.name):
                self.convolution.build([None, None, None, self.in_channels])
        if getattr(self, 'normalization', None) is not None:
            with tf.name_scope(self.normalization.name):
                self.normalization.build([None, None, None, self.out_channels])
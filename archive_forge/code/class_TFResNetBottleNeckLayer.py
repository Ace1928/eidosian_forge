from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_resnet import ResNetConfig
class TFResNetBottleNeckLayer(keras.layers.Layer):
    """
    A classic ResNet's bottleneck layer composed by three `3x3` convolutions.

    The first `1x1` convolution reduces the input by a factor of `reduction` in order to make the second `3x3`
    convolution faster. The last `1x1` convolution remaps the reduced features to `out_channels`.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int=1, activation: str='relu', reduction: int=4, **kwargs) -> None:
        super().__init__(**kwargs)
        should_apply_shortcut = in_channels != out_channels or stride != 1
        reduces_channels = out_channels // reduction
        self.conv0 = TFResNetConvLayer(in_channels, reduces_channels, kernel_size=1, name='layer.0')
        self.conv1 = TFResNetConvLayer(reduces_channels, reduces_channels, stride=stride, name='layer.1')
        self.conv2 = TFResNetConvLayer(reduces_channels, out_channels, kernel_size=1, activation=None, name='layer.2')
        self.shortcut = TFResNetShortCut(in_channels, out_channels, stride=stride, name='shortcut') if should_apply_shortcut else keras.layers.Activation('linear', name='shortcut')
        self.activation = ACT2FN[activation]

    def call(self, hidden_state: tf.Tensor, training: bool=False) -> tf.Tensor:
        residual = hidden_state
        hidden_state = self.conv0(hidden_state, training=training)
        hidden_state = self.conv1(hidden_state, training=training)
        hidden_state = self.conv2(hidden_state, training=training)
        residual = self.shortcut(residual, training=training)
        hidden_state += residual
        hidden_state = self.activation(hidden_state)
        return hidden_state

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'conv0', None) is not None:
            with tf.name_scope(self.conv0.name):
                self.conv0.build(None)
        if getattr(self, 'conv1', None) is not None:
            with tf.name_scope(self.conv1.name):
                self.conv1.build(None)
        if getattr(self, 'conv2', None) is not None:
            with tf.name_scope(self.conv2.name):
                self.conv2.build(None)
        if getattr(self, 'shortcut', None) is not None:
            with tf.name_scope(self.shortcut.name):
                self.shortcut.build(None)
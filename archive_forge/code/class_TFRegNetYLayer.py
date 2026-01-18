from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list
from ...utils import logging
from .configuration_regnet import RegNetConfig
class TFRegNetYLayer(keras.layers.Layer):
    """
    RegNet's Y layer: an X layer with Squeeze and Excitation.
    """

    def __init__(self, config: RegNetConfig, in_channels: int, out_channels: int, stride: int=1, **kwargs):
        super().__init__(**kwargs)
        should_apply_shortcut = in_channels != out_channels or stride != 1
        groups = max(1, out_channels // config.groups_width)
        self.shortcut = TFRegNetShortCut(in_channels, out_channels, stride=stride, name='shortcut') if should_apply_shortcut else keras.layers.Activation('linear', name='shortcut')
        self.layers = [TFRegNetConvLayer(in_channels, out_channels, kernel_size=1, activation=config.hidden_act, name='layer.0'), TFRegNetConvLayer(out_channels, out_channels, stride=stride, groups=groups, activation=config.hidden_act, name='layer.1'), TFRegNetSELayer(out_channels, reduced_channels=int(round(in_channels / 4)), name='layer.2'), TFRegNetConvLayer(out_channels, out_channels, kernel_size=1, activation=None, name='layer.3')]
        self.activation = ACT2FN[config.hidden_act]

    def call(self, hidden_state):
        residual = hidden_state
        for layer_module in self.layers:
            hidden_state = layer_module(hidden_state)
        residual = self.shortcut(residual)
        hidden_state += residual
        hidden_state = self.activation(hidden_state)
        return hidden_state

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'shortcut', None) is not None:
            with tf.name_scope(self.shortcut.name):
                self.shortcut.build(None)
        if getattr(self, 'layers', None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)
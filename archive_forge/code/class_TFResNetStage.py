from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_resnet import ResNetConfig
class TFResNetStage(keras.layers.Layer):
    """
    A ResNet stage composed of stacked layers.
    """

    def __init__(self, config: ResNetConfig, in_channels: int, out_channels: int, stride: int=2, depth: int=2, **kwargs) -> None:
        super().__init__(**kwargs)
        layer = TFResNetBottleNeckLayer if config.layer_type == 'bottleneck' else TFResNetBasicLayer
        layers = [layer(in_channels, out_channels, stride=stride, activation=config.hidden_act, name='layers.0')]
        layers += [layer(out_channels, out_channels, activation=config.hidden_act, name=f'layers.{i + 1}') for i in range(depth - 1)]
        self.stage_layers = layers

    def call(self, hidden_state: tf.Tensor, training: bool=False) -> tf.Tensor:
        for layer in self.stage_layers:
            hidden_state = layer(hidden_state, training=training)
        return hidden_state

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'stage_layers', None) is not None:
            for layer in self.stage_layers:
                with tf.name_scope(layer.name):
                    layer.build(None)
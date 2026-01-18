from __future__ import annotations
from typing import Dict, Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...file_utils import (
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import logging
from .configuration_mobilevit import MobileViTConfig
class TFMobileViTEncoder(keras.layers.Layer):

    def __init__(self, config: MobileViTConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config = config
        self.layers = []
        dilate_layer_4 = dilate_layer_5 = False
        if config.output_stride == 8:
            dilate_layer_4 = True
            dilate_layer_5 = True
        elif config.output_stride == 16:
            dilate_layer_5 = True
        dilation = 1
        layer_1 = TFMobileViTMobileNetLayer(config, in_channels=config.neck_hidden_sizes[0], out_channels=config.neck_hidden_sizes[1], stride=1, num_stages=1, name='layer.0')
        self.layers.append(layer_1)
        layer_2 = TFMobileViTMobileNetLayer(config, in_channels=config.neck_hidden_sizes[1], out_channels=config.neck_hidden_sizes[2], stride=2, num_stages=3, name='layer.1')
        self.layers.append(layer_2)
        layer_3 = TFMobileViTLayer(config, in_channels=config.neck_hidden_sizes[2], out_channels=config.neck_hidden_sizes[3], stride=2, hidden_size=config.hidden_sizes[0], num_stages=2, name='layer.2')
        self.layers.append(layer_3)
        if dilate_layer_4:
            dilation *= 2
        layer_4 = TFMobileViTLayer(config, in_channels=config.neck_hidden_sizes[3], out_channels=config.neck_hidden_sizes[4], stride=2, hidden_size=config.hidden_sizes[1], num_stages=4, dilation=dilation, name='layer.3')
        self.layers.append(layer_4)
        if dilate_layer_5:
            dilation *= 2
        layer_5 = TFMobileViTLayer(config, in_channels=config.neck_hidden_sizes[4], out_channels=config.neck_hidden_sizes[5], stride=2, hidden_size=config.hidden_sizes[2], num_stages=3, dilation=dilation, name='layer.4')
        self.layers.append(layer_5)

    def call(self, hidden_states: tf.Tensor, output_hidden_states: bool=False, return_dict: bool=True, training: bool=False) -> Union[tuple, TFBaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        for i, layer_module in enumerate(self.layers):
            hidden_states = layer_module(hidden_states, training=training)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states] if v is not None))
        return TFBaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'layers', None) is not None:
            for layer_module in self.layers:
                with tf.name_scope(layer_module.name):
                    layer_module.build(None)
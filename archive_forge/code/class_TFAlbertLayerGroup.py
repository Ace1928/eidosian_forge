from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_albert import AlbertConfig
class TFAlbertLayerGroup(keras.layers.Layer):

    def __init__(self, config: AlbertConfig, **kwargs):
        super().__init__(**kwargs)
        self.albert_layers = [TFAlbertLayer(config, name=f'albert_layers_._{i}') for i in range(config.inner_group_num)]

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, output_hidden_states: bool, training: bool=False) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        layer_hidden_states = () if output_hidden_states else None
        layer_attentions = () if output_attentions else None
        for layer_index, albert_layer in enumerate(self.albert_layers):
            if output_hidden_states:
                layer_hidden_states = layer_hidden_states + (hidden_states,)
            layer_output = albert_layer(hidden_states=hidden_states, attention_mask=attention_mask, head_mask=head_mask[layer_index], output_attentions=output_attentions, training=training)
            hidden_states = layer_output[0]
            if output_attentions:
                layer_attentions = layer_attentions + (layer_output[1],)
        if output_hidden_states:
            layer_hidden_states = layer_hidden_states + (hidden_states,)
        return tuple((v for v in [hidden_states, layer_hidden_states, layer_attentions] if v is not None))

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'albert_layers', None) is not None:
            for layer in self.albert_layers:
                with tf.name_scope(layer.name):
                    layer.build(None)
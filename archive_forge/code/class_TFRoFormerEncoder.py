from __future__ import annotations
import math
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_roformer import RoFormerConfig
class TFRoFormerEncoder(keras.layers.Layer):

    def __init__(self, config: RoFormerConfig, **kwargs):
        super().__init__(**kwargs)
        self.embed_positions = TFRoFormerSinusoidalPositionalEmbedding(config.max_position_embeddings, config.hidden_size // config.num_attention_heads, name='embed_positions')
        self.layer = [TFRoFormerLayer(config, name=f'layer_._{i}') for i in range(config.num_hidden_layers)]

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, output_hidden_states: bool, return_dict: bool, training: bool=False) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        sinusoidal_pos = self.embed_positions(shape_list(hidden_states)[:-1])[None, None, :, :]
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states=hidden_states, attention_mask=attention_mask, sinusoidal_pos=sinusoidal_pos, head_mask=head_mask[i], output_attentions=output_attentions, training=training)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None))
        return TFBaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'embed_positions', None) is not None:
            with tf.name_scope(self.embed_positions.name):
                self.embed_positions.build(None)
        if getattr(self, 'layer', None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)
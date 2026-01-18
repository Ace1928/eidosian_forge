from __future__ import annotations
import math
from typing import Dict, Optional, Sequence, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_deberta import DebertaConfig
class TFDebertaEncoder(keras.layers.Layer):

    def __init__(self, config: DebertaConfig, **kwargs):
        super().__init__(**kwargs)
        self.layer = [TFDebertaLayer(config, name=f'layer_._{i}') for i in range(config.num_hidden_layers)]
        self.relative_attention = getattr(config, 'relative_attention', False)
        self.config = config
        if self.relative_attention:
            self.max_relative_positions = getattr(config, 'max_relative_positions', -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if self.relative_attention:
            self.rel_embeddings = self.add_weight(name='rel_embeddings.weight', shape=[self.max_relative_positions * 2, self.config.hidden_size], initializer=get_initializer(self.config.initializer_range))
        if getattr(self, 'layer', None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)

    def get_rel_embedding(self):
        rel_embeddings = self.rel_embeddings if self.relative_attention else None
        return rel_embeddings

    def get_attention_mask(self, attention_mask):
        if len(shape_list(attention_mask)) <= 2:
            extended_attention_mask = tf.expand_dims(tf.expand_dims(attention_mask, 1), 2)
            attention_mask = extended_attention_mask * tf.expand_dims(tf.squeeze(extended_attention_mask, -2), -1)
            attention_mask = tf.cast(attention_mask, tf.uint8)
        elif len(shape_list(attention_mask)) == 3:
            attention_mask = tf.expand_dims(attention_mask, 1)
        return attention_mask

    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        if self.relative_attention and relative_pos is None:
            q = shape_list(query_states)[-2] if query_states is not None else shape_list(hidden_states)[-2]
            relative_pos = build_relative_position(q, shape_list(hidden_states)[-2])
        return relative_pos

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, query_states: tf.Tensor=None, relative_pos: tf.Tensor=None, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True, training: bool=False) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        attention_mask = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)
        if isinstance(hidden_states, Sequence):
            next_kv = hidden_states[0]
        else:
            next_kv = hidden_states
        rel_embeddings = self.get_rel_embedding()
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states=next_kv, attention_mask=attention_mask, query_states=query_states, relative_pos=relative_pos, rel_embeddings=rel_embeddings, output_attentions=output_attentions, training=training)
            hidden_states = layer_outputs[0]
            if query_states is not None:
                query_states = hidden_states
                if isinstance(hidden_states, Sequence):
                    next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
            else:
                next_kv = hidden_states
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None))
        return TFBaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions)
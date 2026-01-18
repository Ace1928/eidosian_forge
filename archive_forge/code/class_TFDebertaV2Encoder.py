from __future__ import annotations
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_deberta_v2 import DebertaV2Config
class TFDebertaV2Encoder(keras.layers.Layer):

    def __init__(self, config: DebertaV2Config, **kwargs):
        super().__init__(**kwargs)
        self.layer = [TFDebertaV2Layer(config, name=f'layer_._{i}') for i in range(config.num_hidden_layers)]
        self.relative_attention = getattr(config, 'relative_attention', False)
        self.config = config
        if self.relative_attention:
            self.max_relative_positions = getattr(config, 'max_relative_positions', -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.position_buckets = getattr(config, 'position_buckets', -1)
            self.pos_ebd_size = self.max_relative_positions * 2
            if self.position_buckets > 0:
                self.pos_ebd_size = self.position_buckets * 2
        self.norm_rel_ebd = [x.strip() for x in getattr(config, 'norm_rel_ebd', 'none').lower().split('|')]
        if 'layer_norm' in self.norm_rel_ebd:
            self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='LayerNorm')
        self.conv = TFDebertaV2ConvLayer(config, name='conv') if getattr(config, 'conv_kernel_size', 0) > 0 else None

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if self.relative_attention:
            self.rel_embeddings = self.add_weight(name='rel_embeddings.weight', shape=[self.pos_ebd_size, self.config.hidden_size], initializer=get_initializer(self.config.initializer_range))
        if getattr(self, 'conv', None) is not None:
            with tf.name_scope(self.conv.name):
                self.conv.build(None)
        if getattr(self, 'LayerNorm', None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, self.config.hidden_size])
        if getattr(self, 'layer', None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)

    def get_rel_embedding(self):
        rel_embeddings = self.rel_embeddings if self.relative_attention else None
        if rel_embeddings is not None and 'layer_norm' in self.norm_rel_ebd:
            rel_embeddings = self.LayerNorm(rel_embeddings)
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
            relative_pos = build_relative_position(q, shape_list(hidden_states)[-2], bucket_size=self.position_buckets, max_position=self.max_relative_positions)
        return relative_pos

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, query_states: tf.Tensor=None, relative_pos: tf.Tensor=None, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True, training: bool=False) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        if len(shape_list(attention_mask)) <= 2:
            input_mask = attention_mask
        else:
            input_mask = tf.cast(tf.math.reduce_sum(attention_mask, axis=-2) > 0, dtype=tf.uint8)
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        attention_mask = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)
        next_kv = hidden_states
        rel_embeddings = self.get_rel_embedding()
        output_states = next_kv
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (output_states,)
            layer_outputs = layer_module(hidden_states=next_kv, attention_mask=attention_mask, query_states=query_states, relative_pos=relative_pos, rel_embeddings=rel_embeddings, output_attentions=output_attentions, training=training)
            output_states = layer_outputs[0]
            if i == 0 and self.conv is not None:
                output_states = self.conv(hidden_states, output_states, input_mask)
            next_kv = output_states
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (output_states,)
        if not return_dict:
            return tuple((v for v in [output_states, all_hidden_states, all_attentions] if v is not None))
        return TFBaseModelOutput(last_hidden_state=output_states, hidden_states=all_hidden_states, attentions=all_attentions)
from __future__ import annotations
import os
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, stable_softmax
from ...utils import logging
from .configuration_esm import EsmConfig
class TFEsmSelfAttention(keras.layers.Layer):

    def __init__(self, config, position_embedding_type=None, name=None):
        super().__init__(name=name)
        if config.hidden_size % config.num_attention_heads != 0 and (not hasattr(config, 'embedding_size')):
            raise ValueError(f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})')
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = keras.layers.Dense(self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='query')
        self.key = keras.layers.Dense(self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='key')
        self.value = keras.layers.Dense(self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='value')
        self.dropout = keras.layers.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(config, 'position_embedding_type', 'absolute')
        self.rotary_embeddings = None
        if self.position_embedding_type == 'relative_key' or self.position_embedding_type == 'relative_key_query':
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = keras.layers.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size, embeddings_initializer=get_initializer(config.initializer_range))
        elif self.position_embedding_type == 'rotary':
            self.rotary_embeddings = TFRotaryEmbedding(dim=self.attention_head_size, name='rotary_embeddings')
        self.is_decoder = config.is_decoder
        self.config = config

    def transpose_for_scores(self, x: tf.Tensor) -> tf.Tensor:
        new_x_shape = shape_list(x)[:-1] + [self.num_attention_heads, self.attention_head_size]
        x = tf.reshape(x, new_x_shape)
        return tf.transpose(x, perm=(0, 2, 1, 3))

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, encoder_hidden_states: tf.Tensor | None=None, encoder_attention_mask: tf.Tensor | None=None, past_key_value: Tuple[Tuple[tf.Tensor]] | None=None, output_attentions: Optional[bool]=False, training: bool=False) -> Tuple[tf.Tensor]:
        mixed_query_layer = self.query(hidden_states)
        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention and past_key_value is not None:
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = tf.concat([past_key_value[0], key_layer], axis=2)
            value_layer = tf.concat([past_key_value[1], value_layer], axis=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        query_layer = query_layer * self.attention_head_size ** (-0.5)
        if self.is_decoder:
            past_key_value = (key_layer, value_layer)
        if self.position_embedding_type == 'rotary':
            query_layer, key_layer = self.rotary_embeddings(query_layer, key_layer)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        if self.position_embedding_type == 'relative_key' or self.position_embedding_type == 'relative_key_query':
            seq_length = shape_list(hidden_states)[1]
            position_ids_l = tf.expand_dims(tf.range(seq_length, dtype=tf.int64), -1)
            position_ids_r = tf.expand_dims(tf.range(seq_length, dtype=tf.int64), 0)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = tf.cast(positional_embedding, query_layer.dtype)
            if self.position_embedding_type == 'relative_key':
                relative_position_scores = tf.einsum('bhld,lrd->bhlr', query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == 'relative_key_query':
                relative_position_scores_query = tf.einsum('bhld,lrd->bhlr', query_layer, positional_embedding)
                relative_position_scores_key = tf.einsum('bhrd,lrd->bhlr', key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = stable_softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs, training=training)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = attention_probs @ value_layer
        context_layer = tf.transpose(context_layer, perm=(0, 2, 1, 3))
        new_context_layer_shape = shape_list(context_layer)[:-2] + [self.all_head_size]
        context_layer = tf.reshape(context_layer, new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'query', None) is not None:
            with tf.name_scope(self.query.name):
                self.query.build([None, None, self.config.hidden_size])
        if getattr(self, 'key', None) is not None:
            with tf.name_scope(self.key.name):
                self.key.build([None, None, self.config.hidden_size])
        if getattr(self, 'value', None) is not None:
            with tf.name_scope(self.value.name):
                self.value.build([None, None, self.config.hidden_size])
        if getattr(self, 'rotary_embeddings', None) is not None:
            with tf.name_scope(self.rotary_embeddings.name):
                self.rotary_embeddings.build(None)
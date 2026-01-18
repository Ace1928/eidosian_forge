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
class TFDebertaV2DisentangledSelfAttention(keras.layers.Layer):
    """
    Disentangled self-attention module

    Parameters:
        config (`DebertaV2Config`):
            A model config class instance with the configuration to build a new model. The schema is similar to
            *BertConfig*, for more details, please refer [`DebertaV2Config`]

    """

    def __init__(self, config: DebertaV2Config, **kwargs):
        super().__init__(**kwargs)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})')
        self.num_attention_heads = config.num_attention_heads
        _attention_head_size = config.hidden_size // config.num_attention_heads
        self.attention_head_size = getattr(config, 'attention_head_size', _attention_head_size)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query_proj = keras.layers.Dense(self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='query_proj', use_bias=True)
        self.key_proj = keras.layers.Dense(self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='key_proj', use_bias=True)
        self.value_proj = keras.layers.Dense(self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='value_proj', use_bias=True)
        self.share_att_key = getattr(config, 'share_att_key', False)
        self.pos_att_type = config.pos_att_type if config.pos_att_type is not None else []
        self.relative_attention = getattr(config, 'relative_attention', False)
        if self.relative_attention:
            self.position_buckets = getattr(config, 'position_buckets', -1)
            self.max_relative_positions = getattr(config, 'max_relative_positions', -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.pos_ebd_size = self.max_relative_positions
            if self.position_buckets > 0:
                self.pos_ebd_size = self.position_buckets
            self.pos_dropout = TFDebertaV2StableDropout(config.hidden_dropout_prob, name='pos_dropout')
            if not self.share_att_key:
                if 'c2p' in self.pos_att_type:
                    self.pos_key_proj = keras.layers.Dense(self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='pos_proj', use_bias=True)
                if 'p2c' in self.pos_att_type:
                    self.pos_query_proj = keras.layers.Dense(self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='pos_q_proj')
        self.softmax = TFDebertaV2XSoftmax(axis=-1)
        self.dropout = TFDebertaV2StableDropout(config.attention_probs_dropout_prob, name='dropout')
        self.config = config

    def transpose_for_scores(self, tensor: tf.Tensor, attention_heads: int) -> tf.Tensor:
        tensor_shape = shape_list(tensor)
        shape = tensor_shape[:-1] + [attention_heads, tensor_shape[-1] // attention_heads]
        tensor = tf.reshape(tensor=tensor, shape=shape)
        tensor = tf.transpose(tensor, perm=[0, 2, 1, 3])
        x_shape = shape_list(tensor)
        tensor = tf.reshape(tensor, shape=[-1, x_shape[-2], x_shape[-1]])
        return tensor

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor, query_states: tf.Tensor=None, relative_pos: tf.Tensor=None, rel_embeddings: tf.Tensor=None, output_attentions: bool=False, training: bool=False) -> Tuple[tf.Tensor]:
        """
        Call the module

        Args:
            hidden_states (`tf.Tensor`):
                Input states to the module usually the output from previous layer, it will be the Q,K and V in
                *Attention(Q,K,V)*

            attention_mask (`tf.Tensor`):
                An attention mask matrix of shape [*B*, *N*, *N*] where *B* is the batch size, *N* is the maximum
                sequence length in which element [i,j] = *1* means the *i* th token in the input can attend to the *j*
                th token.

            return_att (`bool`, optional):
                Whether return the attention matrix.

            query_states (`tf.Tensor`, optional):
                The *Q* state in *Attention(Q,K,V)*.

            relative_pos (`tf.Tensor`):
                The relative position encoding between the tokens in the sequence. It's of shape [*B*, *N*, *N*] with
                values ranging in [*-max_relative_positions*, *max_relative_positions*].

            rel_embeddings (`tf.Tensor`):
                The embedding of relative distances. It's a tensor of shape [\\(2 \\times
                \\text{max_relative_positions}\\), *hidden_size*].


        """
        if query_states is None:
            query_states = hidden_states
        query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        rel_att = None
        scale_factor = 1
        if 'c2p' in self.pos_att_type:
            scale_factor += 1
        if 'p2c' in self.pos_att_type:
            scale_factor += 1
        scale = tf.math.sqrt(tf.cast(shape_list(query_layer)[-1] * scale_factor, tf.float32))
        attention_scores = tf.matmul(query_layer, tf.transpose(key_layer, [0, 2, 1]) / scale)
        if self.relative_attention:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            rel_att = self.disentangled_att_bias(query_layer, key_layer, relative_pos, rel_embeddings, scale_factor)
        if rel_att is not None:
            attention_scores = attention_scores + rel_att
        attention_scores = tf.reshape(attention_scores, (-1, self.num_attention_heads, shape_list(attention_scores)[-2], shape_list(attention_scores)[-1]))
        attention_probs = self.softmax(attention_scores, attention_mask)
        attention_probs = self.dropout(attention_probs, training=training)
        context_layer = tf.matmul(tf.reshape(attention_probs, [-1, shape_list(attention_probs)[-2], shape_list(attention_probs)[-1]]), value_layer)
        context_layer = tf.transpose(tf.reshape(context_layer, [-1, self.num_attention_heads, shape_list(context_layer)[-2], shape_list(context_layer)[-1]]), [0, 2, 1, 3])
        context_layer_shape = shape_list(context_layer)
        new_context_layer_shape = context_layer_shape[:-2] + [context_layer_shape[-2] * context_layer_shape[-1]]
        context_layer = tf.reshape(context_layer, new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

    def disentangled_att_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        if relative_pos is None:
            q = shape_list(query_layer)[-2]
            relative_pos = build_relative_position(q, shape_list(key_layer)[-2], bucket_size=self.position_buckets, max_position=self.max_relative_positions)
        shape_list_pos = shape_list(relative_pos)
        if len(shape_list_pos) == 2:
            relative_pos = tf.expand_dims(tf.expand_dims(relative_pos, 0), 0)
        elif len(shape_list_pos) == 3:
            relative_pos = tf.expand_dims(relative_pos, 1)
        elif len(shape_list_pos) != 4:
            raise ValueError(f'Relative position ids must be of dim 2 or 3 or 4. {len(shape_list_pos)}')
        att_span = self.pos_ebd_size
        rel_embeddings = tf.expand_dims(rel_embeddings[self.pos_ebd_size - att_span:self.pos_ebd_size + att_span, :], 0)
        if self.share_att_key:
            pos_query_layer = tf.tile(self.transpose_for_scores(self.query_proj(rel_embeddings), self.num_attention_heads), [shape_list(query_layer)[0] // self.num_attention_heads, 1, 1])
            pos_key_layer = tf.tile(self.transpose_for_scores(self.key_proj(rel_embeddings), self.num_attention_heads), [shape_list(query_layer)[0] // self.num_attention_heads, 1, 1])
        else:
            if 'c2p' in self.pos_att_type:
                pos_key_layer = tf.tile(self.transpose_for_scores(self.pos_key_proj(rel_embeddings), self.num_attention_heads), [shape_list(query_layer)[0] // self.num_attention_heads, 1, 1])
            if 'p2c' in self.pos_att_type:
                pos_query_layer = tf.tile(self.transpose_for_scores(self.pos_query_proj(rel_embeddings), self.num_attention_heads), [shape_list(query_layer)[0] // self.num_attention_heads, 1, 1])
        score = 0
        if 'c2p' in self.pos_att_type:
            scale = tf.math.sqrt(tf.cast(shape_list(pos_key_layer)[-1] * scale_factor, tf.float32))
            c2p_att = tf.matmul(query_layer, tf.transpose(pos_key_layer, [0, 2, 1]))
            c2p_pos = tf.clip_by_value(relative_pos + att_span, 0, att_span * 2 - 1)
            c2p_att = take_along_axis(c2p_att, tf.broadcast_to(tf.squeeze(c2p_pos, 0), [shape_list(query_layer)[0], shape_list(query_layer)[1], shape_list(relative_pos)[-1]]))
            score += c2p_att / scale
        if 'p2c' in self.pos_att_type:
            scale = tf.math.sqrt(tf.cast(shape_list(pos_query_layer)[-1] * scale_factor, tf.float32))
            if shape_list(key_layer)[-2] != shape_list(query_layer)[-2]:
                r_pos = build_relative_position(shape_list(key_layer)[-2], shape_list(key_layer)[-2], bucket_size=self.position_buckets, max_position=self.max_relative_positions)
                r_pos = tf.expand_dims(r_pos, 0)
            else:
                r_pos = relative_pos
            p2c_pos = tf.clip_by_value(-r_pos + att_span, 0, att_span * 2 - 1)
            p2c_att = tf.matmul(key_layer, tf.transpose(pos_query_layer, [0, 2, 1]))
            p2c_att = tf.transpose(take_along_axis(p2c_att, tf.broadcast_to(tf.squeeze(p2c_pos, 0), [shape_list(query_layer)[0], shape_list(key_layer)[-2], shape_list(key_layer)[-2]])), [0, 2, 1])
            score += p2c_att / scale
        return score

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'query_proj', None) is not None:
            with tf.name_scope(self.query_proj.name):
                self.query_proj.build([None, None, self.config.hidden_size])
        if getattr(self, 'key_proj', None) is not None:
            with tf.name_scope(self.key_proj.name):
                self.key_proj.build([None, None, self.config.hidden_size])
        if getattr(self, 'value_proj', None) is not None:
            with tf.name_scope(self.value_proj.name):
                self.value_proj.build([None, None, self.config.hidden_size])
        if getattr(self, 'dropout', None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)
        if getattr(self, 'pos_dropout', None) is not None:
            with tf.name_scope(self.pos_dropout.name):
                self.pos_dropout.build(None)
        if getattr(self, 'pos_key_proj', None) is not None:
            with tf.name_scope(self.pos_key_proj.name):
                self.pos_key_proj.build([None, None, self.config.hidden_size])
        if getattr(self, 'pos_query_proj', None) is not None:
            with tf.name_scope(self.pos_query_proj.name):
                self.pos_query_proj.build([None, None, self.config.hidden_size])
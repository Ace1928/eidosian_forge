from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_funnel import FunnelConfig
class TFFunnelRelMultiheadAttention(keras.layers.Layer):

    def __init__(self, config, block_index, **kwargs):
        super().__init__(**kwargs)
        self.attention_type = config.attention_type
        self.n_head = n_head = config.n_head
        self.d_head = d_head = config.d_head
        self.d_model = d_model = config.d_model
        self.initializer_range = config.initializer_range
        self.block_index = block_index
        self.hidden_dropout = keras.layers.Dropout(config.hidden_dropout)
        self.attention_dropout = keras.layers.Dropout(config.attention_dropout)
        initializer = get_initializer(config.initializer_range)
        self.q_head = keras.layers.Dense(n_head * d_head, use_bias=False, kernel_initializer=initializer, name='q_head')
        self.k_head = keras.layers.Dense(n_head * d_head, kernel_initializer=initializer, name='k_head')
        self.v_head = keras.layers.Dense(n_head * d_head, kernel_initializer=initializer, name='v_head')
        self.post_proj = keras.layers.Dense(d_model, kernel_initializer=initializer, name='post_proj')
        self.layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layer_norm')
        self.scale = 1.0 / d_head ** 0.5

    def build(self, input_shape=None):
        n_head, d_head, d_model = (self.n_head, self.d_head, self.d_model)
        initializer = get_initializer(self.initializer_range)
        self.r_w_bias = self.add_weight(shape=(n_head, d_head), initializer=initializer, trainable=True, name='r_w_bias')
        self.r_r_bias = self.add_weight(shape=(n_head, d_head), initializer=initializer, trainable=True, name='r_r_bias')
        self.r_kernel = self.add_weight(shape=(d_model, n_head, d_head), initializer=initializer, trainable=True, name='r_kernel')
        self.r_s_bias = self.add_weight(shape=(n_head, d_head), initializer=initializer, trainable=True, name='r_s_bias')
        self.seg_embed = self.add_weight(shape=(2, n_head, d_head), initializer=initializer, trainable=True, name='seg_embed')
        if self.built:
            return
        self.built = True
        if getattr(self, 'q_head', None) is not None:
            with tf.name_scope(self.q_head.name):
                self.q_head.build([None, None, d_model])
        if getattr(self, 'k_head', None) is not None:
            with tf.name_scope(self.k_head.name):
                self.k_head.build([None, None, d_model])
        if getattr(self, 'v_head', None) is not None:
            with tf.name_scope(self.v_head.name):
                self.v_head.build([None, None, d_model])
        if getattr(self, 'post_proj', None) is not None:
            with tf.name_scope(self.post_proj.name):
                self.post_proj.build([None, None, n_head * d_head])
        if getattr(self, 'layer_norm', None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, d_model])

    def relative_positional_attention(self, position_embeds, q_head, context_len, cls_mask=None):
        """Relative attention score for the positional encodings"""
        if self.attention_type == 'factorized':
            phi, pi, psi, omega = position_embeds
            u = self.r_r_bias * self.scale
            w_r = self.r_kernel
            q_r_attention = tf.einsum('binh,dnh->bind', q_head + u, w_r)
            q_r_attention_1 = q_r_attention * phi[:, None]
            q_r_attention_2 = q_r_attention * pi[:, None]
            positional_attn = tf.einsum('bind,jd->bnij', q_r_attention_1, psi) + tf.einsum('bind,jd->bnij', q_r_attention_2, omega)
        else:
            if shape_list(q_head)[1] != context_len:
                shift = 2
                r = position_embeds[self.block_index][1]
            else:
                shift = 1
                r = position_embeds[self.block_index][0]
            v = self.r_r_bias * self.scale
            w_r = self.r_kernel
            r_head = tf.einsum('td,dnh->tnh', r, w_r)
            positional_attn = tf.einsum('binh,tnh->bnit', q_head + v, r_head)
            positional_attn = _relative_shift_gather(positional_attn, context_len, shift)
        if cls_mask is not None:
            positional_attn *= cls_mask
        return positional_attn

    def relative_token_type_attention(self, token_type_mat, q_head, cls_mask=None):
        """Relative attention score for the token_type_ids"""
        if token_type_mat is None:
            return 0
        batch_size, seq_len, context_len = shape_list(token_type_mat)
        r_s_bias = self.r_s_bias * self.scale
        token_type_bias = tf.einsum('bind,snd->bnis', q_head + r_s_bias, self.seg_embed)
        token_type_mat = tf.tile(token_type_mat[:, None], [1, shape_list(q_head)[2], 1, 1])
        diff_token_type, same_token_type = tf.split(token_type_bias, 2, axis=-1)
        token_type_attn = tf.where(token_type_mat, tf.tile(same_token_type, [1, 1, 1, context_len]), tf.tile(diff_token_type, [1, 1, 1, context_len]))
        if cls_mask is not None:
            token_type_attn *= cls_mask
        return token_type_attn

    def call(self, query, key, value, attention_inputs, output_attentions=False, training=False):
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs
        batch_size, seq_len, _ = shape_list(query)
        context_len = shape_list(key)[1]
        n_head, d_head = (self.n_head, self.d_head)
        q_head = tf.reshape(self.q_head(query), [batch_size, seq_len, n_head, d_head])
        k_head = tf.reshape(self.k_head(key), [batch_size, context_len, n_head, d_head])
        v_head = tf.reshape(self.v_head(value), [batch_size, context_len, n_head, d_head])
        q_head = q_head * self.scale
        r_w_bias = self.r_w_bias * self.scale
        content_score = tf.einsum('bind,bjnd->bnij', q_head + r_w_bias, k_head)
        positional_attn = self.relative_positional_attention(position_embeds, q_head, context_len, cls_mask)
        token_type_attn = self.relative_token_type_attention(token_type_mat, q_head, cls_mask)
        attn_score = content_score + positional_attn + token_type_attn
        if attention_mask is not None:
            attention_mask = tf.cast(attention_mask, dtype=attn_score.dtype)
            attn_score = attn_score - INF * (1 - attention_mask[:, None, None])
        attn_prob = stable_softmax(attn_score, axis=-1)
        attn_prob = self.attention_dropout(attn_prob, training=training)
        attn_vec = tf.einsum('bnij,bjnd->bind', attn_prob, v_head)
        attn_out = self.post_proj(tf.reshape(attn_vec, [batch_size, seq_len, n_head * d_head]))
        attn_out = self.hidden_dropout(attn_out, training=training)
        output = self.layer_norm(query + attn_out)
        return (output, attn_prob) if output_attentions else (output,)
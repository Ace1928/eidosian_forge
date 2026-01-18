from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_xlnet import XLNetConfig
class TFXLNetRelativeAttention(keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        if config.d_model % config.n_head != 0:
            raise ValueError(f'The hidden size ({config.d_model}) is not a multiple of the number of attention heads ({config.n_head}')
        self.n_head = config.n_head
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.scale = 1 / config.d_head ** 0.5
        self.initializer_range = config.initializer_range
        self.output_attentions = config.output_attentions
        self.layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layer_norm')
        self.dropout = keras.layers.Dropout(config.dropout)
        self.config = config

    def build(self, input_shape=None):
        initializer = get_initializer(self.initializer_range)
        self.q = self.add_weight(shape=(self.d_model, self.n_head, self.d_head), initializer=initializer, trainable=True, name='q')
        self.k = self.add_weight(shape=(self.d_model, self.n_head, self.d_head), initializer=initializer, trainable=True, name='k')
        self.v = self.add_weight(shape=(self.d_model, self.n_head, self.d_head), initializer=initializer, trainable=True, name='v')
        self.o = self.add_weight(shape=(self.d_model, self.n_head, self.d_head), initializer=initializer, trainable=True, name='o')
        self.r = self.add_weight(shape=(self.d_model, self.n_head, self.d_head), initializer=initializer, trainable=True, name='r')
        self.r_r_bias = self.add_weight(shape=(self.n_head, self.d_head), initializer='zeros', trainable=True, name='r_r_bias')
        self.r_s_bias = self.add_weight(shape=(self.n_head, self.d_head), initializer='zeros', trainable=True, name='r_s_bias')
        self.r_w_bias = self.add_weight(shape=(self.n_head, self.d_head), initializer='zeros', trainable=True, name='r_w_bias')
        self.seg_embed = self.add_weight(shape=(2, self.n_head, self.d_head), initializer=initializer, trainable=True, name='seg_embed')
        if self.built:
            return
        self.built = True
        if getattr(self, 'layer_norm', None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.d_model])

    def prune_heads(self, heads):
        raise NotImplementedError

    def rel_shift(self, x, klen=-1):
        """perform relative shift to form the relative attention score."""
        x_size = shape_list(x)
        x = tf.reshape(x, (x_size[1], x_size[0], x_size[2], x_size[3]))
        x = x[1:, ...]
        x = tf.reshape(x, (x_size[0], x_size[1] - 1, x_size[2], x_size[3]))
        x = x[:, 0:klen, :, :]
        return x

    def rel_attn_core(self, q_head, k_head_h, v_head_h, k_head_r, seg_mat, attn_mask, head_mask, output_attentions, training=False):
        """Core relative positional attention operations."""
        ac = tf.einsum('ibnd,jbnd->ijbn', q_head + self.r_w_bias, k_head_h)
        bd = tf.einsum('ibnd,jbnd->ijbn', q_head + self.r_r_bias, k_head_r)
        bd = self.rel_shift(bd, klen=shape_list(ac)[1])
        if seg_mat is None:
            ef = 0
        else:
            ef = tf.einsum('ibnd,snd->ibns', q_head + self.r_s_bias, self.seg_embed)
            ef = tf.einsum('ijbs,ibns->ijbn', seg_mat, ef)
        attn_score = (ac + bd + ef) * self.scale
        if attn_mask is not None:
            if attn_mask.dtype == tf.float16 or attn_mask.dtype == tf.bfloat16:
                attn_score = attn_score - 65500 * attn_mask
            else:
                attn_score = attn_score - 1e+30 * attn_mask
        attn_prob = stable_softmax(attn_score, axis=1)
        attn_prob = self.dropout(attn_prob, training=training)
        if head_mask is not None:
            attn_prob = attn_prob * head_mask
        attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, v_head_h)
        if output_attentions:
            return (attn_vec, attn_prob)
        return attn_vec

    def post_attention(self, h, attn_vec, residual=True, training=False):
        """Post-attention processing."""
        attn_out = tf.einsum('ibnd,hnd->ibh', attn_vec, self.o)
        attn_out = self.dropout(attn_out, training=training)
        if residual:
            attn_out = attn_out + h
        output = self.layer_norm(attn_out)
        return output

    def call(self, h, g, attn_mask_h, attn_mask_g, r, seg_mat, mems: np.ndarray | tf.Tensor | None=None, target_mapping: np.ndarray | tf.Tensor | None=None, head_mask: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=False, training: bool=False):
        if g is not None:
            if mems is not None and len(shape_list(mems)) > 1:
                cat = tf.concat([mems, h], axis=0)
            else:
                cat = h
            k_head_h = tf.einsum('ibh,hnd->ibnd', cat, self.k)
            v_head_h = tf.einsum('ibh,hnd->ibnd', cat, self.v)
            k_head_r = tf.einsum('ibh,hnd->ibnd', r, self.r)
            q_head_h = tf.einsum('ibh,hnd->ibnd', h, self.q)
            attn_vec_h = self.rel_attn_core(q_head_h, k_head_h, v_head_h, k_head_r, seg_mat, attn_mask_h, head_mask, output_attentions, training=training)
            if output_attentions:
                attn_vec_h, attn_prob_h = attn_vec_h
            output_h = self.post_attention(h, attn_vec_h, training=training)
            q_head_g = tf.einsum('ibh,hnd->ibnd', g, self.q)
            if target_mapping is not None:
                q_head_g = tf.einsum('mbnd,mlb->lbnd', q_head_g, target_mapping)
                attn_vec_g = self.rel_attn_core(q_head_g, k_head_h, v_head_h, k_head_r, seg_mat, attn_mask_g, head_mask, output_attentions, training=training)
                if output_attentions:
                    attn_vec_g, attn_prob_g = attn_vec_g
                attn_vec_g = tf.einsum('lbnd,mlb->mbnd', attn_vec_g, target_mapping)
            else:
                attn_vec_g = self.rel_attn_core(q_head_g, k_head_h, v_head_h, k_head_r, seg_mat, attn_mask_g, head_mask, output_attentions, training=training)
                if output_attentions:
                    attn_vec_g, attn_prob_g = attn_vec_g
            output_g = self.post_attention(g, attn_vec_g, training=training)
            if output_attentions:
                attn_prob = (attn_prob_h, attn_prob_g)
        else:
            if mems is not None and len(shape_list(mems)) > 1:
                cat = tf.concat([mems, h], axis=0)
            else:
                cat = h
            q_head_h = tf.einsum('ibh,hnd->ibnd', h, self.q)
            k_head_h = tf.einsum('ibh,hnd->ibnd', cat, self.k)
            v_head_h = tf.einsum('ibh,hnd->ibnd', cat, self.v)
            k_head_r = tf.einsum('ibh,hnd->ibnd', r, self.r)
            attn_vec = self.rel_attn_core(q_head_h, k_head_h, v_head_h, k_head_r, seg_mat, attn_mask_h, head_mask, output_attentions, training=training)
            if output_attentions:
                attn_vec, attn_prob = attn_vec
            output_h = self.post_attention(h, attn_vec, training=training)
            output_g = None
        outputs = (output_h, output_g)
        if output_attentions:
            outputs = outputs + (attn_prob,)
        return outputs
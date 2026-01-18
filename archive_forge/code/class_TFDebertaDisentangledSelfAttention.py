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
class TFDebertaDisentangledSelfAttention(keras.layers.Layer):
    """
    Disentangled self-attention module

    Parameters:
        config (`str`):
            A model config class instance with the configuration to build a new model. The schema is similar to
            *BertConfig*, for more details, please refer [`DebertaConfig`]

    """

    def __init__(self, config: DebertaConfig, **kwargs):
        super().__init__(**kwargs)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})')
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.in_proj = keras.layers.Dense(self.all_head_size * 3, kernel_initializer=get_initializer(config.initializer_range), name='in_proj', use_bias=False)
        self.pos_att_type = config.pos_att_type if config.pos_att_type is not None else []
        self.relative_attention = getattr(config, 'relative_attention', False)
        self.talking_head = getattr(config, 'talking_head', False)
        if self.talking_head:
            self.head_logits_proj = keras.layers.Dense(self.num_attention_heads, kernel_initializer=get_initializer(config.initializer_range), name='head_logits_proj', use_bias=False)
            self.head_weights_proj = keras.layers.Dense(self.num_attention_heads, kernel_initializer=get_initializer(config.initializer_range), name='head_weights_proj', use_bias=False)
        self.softmax = TFDebertaXSoftmax(axis=-1)
        if self.relative_attention:
            self.max_relative_positions = getattr(config, 'max_relative_positions', -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.pos_dropout = TFDebertaStableDropout(config.hidden_dropout_prob, name='pos_dropout')
            if 'c2p' in self.pos_att_type:
                self.pos_proj = keras.layers.Dense(self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='pos_proj', use_bias=False)
            if 'p2c' in self.pos_att_type:
                self.pos_q_proj = keras.layers.Dense(self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='pos_q_proj')
        self.dropout = TFDebertaStableDropout(config.attention_probs_dropout_prob, name='dropout')
        self.config = config

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        self.q_bias = self.add_weight(name='q_bias', shape=self.all_head_size, initializer=keras.initializers.Zeros())
        self.v_bias = self.add_weight(name='v_bias', shape=self.all_head_size, initializer=keras.initializers.Zeros())
        if getattr(self, 'in_proj', None) is not None:
            with tf.name_scope(self.in_proj.name):
                self.in_proj.build([None, None, self.config.hidden_size])
        if getattr(self, 'dropout', None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)
        if getattr(self, 'head_logits_proj', None) is not None:
            with tf.name_scope(self.head_logits_proj.name):
                self.head_logits_proj.build(None)
        if getattr(self, 'head_weights_proj', None) is not None:
            with tf.name_scope(self.head_weights_proj.name):
                self.head_weights_proj.build(None)
        if getattr(self, 'pos_dropout', None) is not None:
            with tf.name_scope(self.pos_dropout.name):
                self.pos_dropout.build(None)
        if getattr(self, 'pos_proj', None) is not None:
            with tf.name_scope(self.pos_proj.name):
                self.pos_proj.build([self.config.hidden_size])
        if getattr(self, 'pos_q_proj', None) is not None:
            with tf.name_scope(self.pos_q_proj.name):
                self.pos_q_proj.build([self.config.hidden_size])

    def transpose_for_scores(self, tensor: tf.Tensor) -> tf.Tensor:
        shape = shape_list(tensor)[:-1] + [self.num_attention_heads, -1]
        tensor = tf.reshape(tensor=tensor, shape=shape)
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

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
            qp = self.in_proj(hidden_states)
            query_layer, key_layer, value_layer = tf.split(self.transpose_for_scores(qp), num_or_size_splits=3, axis=-1)
        else:

            def linear(w, b, x):
                out = tf.matmul(x, w, transpose_b=True)
                if b is not None:
                    out += tf.transpose(b)
                return out
            ws = tf.split(tf.transpose(self.in_proj.weight[0]), num_or_size_splits=self.num_attention_heads * 3, axis=0)
            qkvw = tf.TensorArray(dtype=tf.float32, size=3)
            for k in tf.range(3):
                qkvw_inside = tf.TensorArray(dtype=tf.float32, size=self.num_attention_heads)
                for i in tf.range(self.num_attention_heads):
                    qkvw_inside = qkvw_inside.write(i, ws[i * 3 + k])
                qkvw = qkvw.write(k, qkvw_inside.concat())
            qkvb = [None] * 3
            q = linear(qkvw[0], qkvb[0], query_states)
            k = linear(qkvw[1], qkvb[1], hidden_states)
            v = linear(qkvw[2], qkvb[2], hidden_states)
            query_layer = self.transpose_for_scores(q)
            key_layer = self.transpose_for_scores(k)
            value_layer = self.transpose_for_scores(v)
        query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
        value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
        rel_att = None
        scale_factor = 1 + len(self.pos_att_type)
        scale = math.sqrt(shape_list(query_layer)[-1] * scale_factor)
        query_layer = query_layer / scale
        attention_scores = tf.matmul(query_layer, tf.transpose(key_layer, [0, 1, 3, 2]))
        if self.relative_attention:
            rel_embeddings = self.pos_dropout(rel_embeddings, training=training)
            rel_att = self.disentangled_att_bias(query_layer, key_layer, relative_pos, rel_embeddings, scale_factor)
        if rel_att is not None:
            attention_scores = attention_scores + rel_att
        if self.talking_head:
            attention_scores = tf.transpose(self.head_logits_proj(tf.transpose(attention_scores, [0, 2, 3, 1])), [0, 3, 1, 2])
        attention_probs = self.softmax(attention_scores, attention_mask)
        attention_probs = self.dropout(attention_probs, training=training)
        if self.talking_head:
            attention_probs = tf.transpose(self.head_weights_proj(tf.transpose(attention_probs, [0, 2, 3, 1])), [0, 3, 1, 2])
        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
        context_layer_shape = shape_list(context_layer)
        new_context_layer_shape = context_layer_shape[:-2] + [context_layer_shape[-2] * context_layer_shape[-1]]
        context_layer = tf.reshape(context_layer, new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

    def disentangled_att_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        if relative_pos is None:
            q = shape_list(query_layer)[-2]
            relative_pos = build_relative_position(q, shape_list(key_layer)[-2])
        shape_list_pos = shape_list(relative_pos)
        if len(shape_list_pos) == 2:
            relative_pos = tf.expand_dims(tf.expand_dims(relative_pos, 0), 0)
        elif len(shape_list_pos) == 3:
            relative_pos = tf.expand_dims(relative_pos, 1)
        elif len(shape_list_pos) != 4:
            raise ValueError(f'Relative position ids must be of dim 2 or 3 or 4. {len(shape_list_pos)}')
        att_span = tf.cast(tf.minimum(tf.maximum(shape_list(query_layer)[-2], shape_list(key_layer)[-2]), self.max_relative_positions), tf.int64)
        rel_embeddings = tf.expand_dims(rel_embeddings[self.max_relative_positions - att_span:self.max_relative_positions + att_span, :], 0)
        score = 0
        if 'c2p' in self.pos_att_type:
            pos_key_layer = self.pos_proj(rel_embeddings)
            pos_key_layer = self.transpose_for_scores(pos_key_layer)
            c2p_att = tf.matmul(query_layer, tf.transpose(pos_key_layer, [0, 1, 3, 2]))
            c2p_pos = tf.clip_by_value(relative_pos + att_span, 0, att_span * 2 - 1)
            c2p_att = torch_gather(c2p_att, c2p_dynamic_expand(c2p_pos, query_layer, relative_pos), -1)
            score += c2p_att
        if 'p2c' in self.pos_att_type:
            pos_query_layer = self.pos_q_proj(rel_embeddings)
            pos_query_layer = self.transpose_for_scores(pos_query_layer)
            pos_query_layer /= tf.math.sqrt(tf.cast(shape_list(pos_query_layer)[-1] * scale_factor, dtype=tf.float32))
            if shape_list(query_layer)[-2] != shape_list(key_layer)[-2]:
                r_pos = build_relative_position(shape_list(key_layer)[-2], shape_list(key_layer)[-2])
            else:
                r_pos = relative_pos
            p2c_pos = tf.clip_by_value(-r_pos + att_span, 0, att_span * 2 - 1)
            p2c_att = tf.matmul(key_layer, tf.transpose(pos_query_layer, [0, 1, 3, 2]))
            p2c_att = tf.transpose(torch_gather(p2c_att, p2c_dynamic_expand(p2c_pos, query_layer, key_layer), -1), [0, 1, 3, 2])
            if shape_list(query_layer)[-2] != shape_list(key_layer)[-2]:
                pos_index = tf.expand_dims(relative_pos[:, :, :, 0], -1)
                p2c_att = torch_gather(p2c_att, pos_dynamic_expand(pos_index, p2c_att, key_layer), -2)
            score += p2c_att
        return score
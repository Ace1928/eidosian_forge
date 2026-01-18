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
class TFFunnelAttentionStructure:
    """
    Contains helpers for `TFFunnelRelMultiheadAttention `.
    """
    cls_token_type_id: int = 2

    def __init__(self, config):
        self.d_model = config.d_model
        self.attention_type = config.attention_type
        self.num_blocks = config.num_blocks
        self.separate_cls = config.separate_cls
        self.truncate_seq = config.truncate_seq
        self.pool_q_only = config.pool_q_only
        self.pooling_type = config.pooling_type
        self.sin_dropout = keras.layers.Dropout(config.hidden_dropout)
        self.cos_dropout = keras.layers.Dropout(config.hidden_dropout)
        self.pooling_mult = None

    def init_attention_inputs(self, inputs_embeds, attention_mask=None, token_type_ids=None, training=False):
        """Returns the attention inputs associated to the inputs of the model."""
        self.pooling_mult = 1
        self.seq_len = seq_len = shape_list(inputs_embeds)[1]
        position_embeds = self.get_position_embeds(seq_len, training=training)
        token_type_mat = self.token_type_ids_to_mat(token_type_ids) if token_type_ids is not None else None
        cls_mask = tf.pad(tf.ones([seq_len - 1, seq_len - 1], dtype=inputs_embeds.dtype), [[1, 0], [1, 0]]) if self.separate_cls else None
        return (position_embeds, token_type_mat, attention_mask, cls_mask)

    def token_type_ids_to_mat(self, token_type_ids):
        """Convert `token_type_ids` to `token_type_mat`."""
        token_type_mat = tf.equal(tf.expand_dims(token_type_ids, -1), tf.expand_dims(token_type_ids, -2))
        cls_ids = tf.equal(token_type_ids, tf.constant([self.cls_token_type_id], dtype=token_type_ids.dtype))
        cls_mat = tf.logical_or(tf.expand_dims(cls_ids, -1), tf.expand_dims(cls_ids, -2))
        return tf.logical_or(cls_mat, token_type_mat)

    def get_position_embeds(self, seq_len, training=False):
        """
        Create and cache inputs related to relative position encoding. Those are very different depending on whether we
        are using the factorized or the relative shift attention:

        For the factorized attention, it returns the matrices (phi, pi, psi, omega) used in the paper, appendix A.2.2,
        final formula.

        For the relative shift attention, it returns all possible vectors R used in the paper, appendix A.2.1, final
        formula.

        Paper link: https://arxiv.org/abs/2006.03236
        """
        if self.attention_type == 'factorized':
            pos_seq = tf.range(0, seq_len, 1.0)
            freq_seq = tf.range(0, self.d_model // 2, 1.0)
            inv_freq = 1 / 10000 ** (freq_seq / (self.d_model // 2))
            sinusoid = tf.einsum('i,d->id', pos_seq, inv_freq)
            sin_embed = tf.sin(sinusoid)
            sin_embed_d = self.sin_dropout(sin_embed, training=training)
            cos_embed = tf.cos(sinusoid)
            cos_embed_d = self.cos_dropout(cos_embed, training=training)
            phi = tf.concat([sin_embed_d, sin_embed_d], axis=-1)
            psi = tf.concat([cos_embed, sin_embed], axis=-1)
            pi = tf.concat([cos_embed_d, cos_embed_d], axis=-1)
            omega = tf.concat([-sin_embed, cos_embed], axis=-1)
            return (phi, pi, psi, omega)
        else:
            freq_seq = tf.range(0, self.d_model // 2, 1.0)
            inv_freq = 1 / 10000 ** (freq_seq / (self.d_model // 2))
            rel_pos_id = tf.range(-seq_len * 2, seq_len * 2, 1.0)
            zero_offset = seq_len * tf.constant(2)
            sinusoid = tf.einsum('i,d->id', rel_pos_id, inv_freq)
            sin_embed = self.sin_dropout(tf.sin(sinusoid), training=training)
            cos_embed = self.cos_dropout(tf.cos(sinusoid), training=training)
            pos_embed = tf.concat([sin_embed, cos_embed], axis=-1)
            pos = tf.range(0, seq_len)
            pooled_pos = pos
            position_embeds_list = []
            for block_index in range(0, self.num_blocks):
                position_embeds_pooling = tf.fill([1], value=-1.0)
                if block_index != 0:
                    pooled_pos = self.stride_pool_pos(pos, block_index)
                    stride = 2 ** (block_index - 1)
                    rel_pos = self.relative_pos(pos, stride, pooled_pos, shift=2)
                    rel_pos = tf.cast(rel_pos, dtype=zero_offset.dtype)
                    rel_pos = rel_pos + zero_offset
                    position_embeds_pooling = tf.gather(pos_embed, rel_pos, axis=0)
                pos = pooled_pos
                stride = 2 ** block_index
                rel_pos = self.relative_pos(pos, stride)
                rel_pos = tf.cast(rel_pos, dtype=zero_offset.dtype)
                rel_pos = rel_pos + zero_offset
                tf.debugging.assert_less(rel_pos, tf.shape(pos_embed)[0])
                position_embeds_no_pooling = tf.gather(pos_embed, rel_pos, axis=0)
                position_embeds_list.append([position_embeds_no_pooling, position_embeds_pooling])
            return position_embeds_list

    def stride_pool_pos(self, pos_id, block_index):
        """
        Pool `pos_id` while keeping the cls token separate (if `self.separate_cls=True`).
        """
        if self.separate_cls:
            cls_pos = tf.constant([-2 ** block_index + 1], dtype=pos_id.dtype)
            pooled_pos_id = pos_id[1:-1] if self.truncate_seq else pos_id[1:]
            return tf.concat([cls_pos, pooled_pos_id[::2]], 0)
        else:
            return pos_id[::2]

    def relative_pos(self, pos, stride, pooled_pos=None, shift=1):
        """
        Build the relative positional vector between `pos` and `pooled_pos`.
        """
        if pooled_pos is None:
            pooled_pos = pos
        ref_point = pooled_pos[0] - pos[0]
        num_remove = shift * shape_list(pooled_pos)[0]
        max_dist = ref_point + num_remove * stride
        min_dist = pooled_pos[0] - pos[-1]
        return tf.range(max_dist, min_dist - 1, -stride)

    def stride_pool(self, tensor, axis):
        """
        Perform pooling by stride slicing the tensor along the given axis.
        """
        if tensor is None:
            return None
        if isinstance(axis, (list, tuple)):
            for ax in axis:
                tensor = self.stride_pool(tensor, ax)
            return tensor
        if isinstance(tensor, (tuple, list)):
            return type(tensor)((self.stride_pool(x, axis) for x in tensor))
        axis %= len(shape_list(tensor))
        axis_slice = slice(None, -1, 2) if self.separate_cls and self.truncate_seq else slice(None, None, 2)
        enc_slice = [slice(None)] * axis + [axis_slice]
        if self.separate_cls:
            cls_slice = [slice(None)] * axis + [slice(None, 1)]
            tensor = tf.concat([tensor[cls_slice], tensor], axis)
        return tensor[enc_slice]

    def pool_tensor(self, tensor, mode='mean', stride=2):
        """Apply 1D pooling to a tensor of size [B x T (x H)]."""
        if tensor is None:
            return None
        if isinstance(tensor, (tuple, list)):
            return type(tensor)((self.pool_tensor(tensor, mode=mode, stride=stride) for x in tensor))
        if self.separate_cls:
            suffix = tensor[:, :-1] if self.truncate_seq else tensor
            tensor = tf.concat([tensor[:, :1], suffix], axis=1)
        ndim = len(shape_list(tensor))
        if ndim == 2:
            tensor = tensor[:, :, None]
        if mode == 'mean':
            tensor = tf.nn.avg_pool1d(tensor, stride, strides=stride, data_format='NWC', padding='SAME')
        elif mode == 'max':
            tensor = tf.nn.max_pool1d(tensor, stride, strides=stride, data_format='NWC', padding='SAME')
        elif mode == 'min':
            tensor = -tf.nn.max_pool1d(-tensor, stride, strides=stride, data_format='NWC', padding='SAME')
        else:
            raise NotImplementedError("The supported modes are 'mean', 'max' and 'min'.")
        return tf.squeeze(tensor, 2) if ndim == 2 else tensor

    def pre_attention_pooling(self, output, attention_inputs):
        """Pool `output` and the proper parts of `attention_inputs` before the attention layer."""
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs
        if self.pool_q_only:
            if self.attention_type == 'factorized':
                position_embeds = self.stride_pool(position_embeds[:2], 0) + position_embeds[2:]
            token_type_mat = self.stride_pool(token_type_mat, 1)
            cls_mask = self.stride_pool(cls_mask, 0)
            output = self.pool_tensor(output, mode=self.pooling_type)
        else:
            self.pooling_mult *= 2
            if self.attention_type == 'factorized':
                position_embeds = self.stride_pool(position_embeds, 0)
            token_type_mat = self.stride_pool(token_type_mat, [1, 2])
            cls_mask = self.stride_pool(cls_mask, [1, 2])
            attention_mask = self.pool_tensor(attention_mask, mode='min')
            output = self.pool_tensor(output, mode=self.pooling_type)
        attention_inputs = (position_embeds, token_type_mat, attention_mask, cls_mask)
        return (output, attention_inputs)

    def post_attention_pooling(self, attention_inputs):
        """Pool the proper parts of `attention_inputs` after the attention layer."""
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs
        if self.pool_q_only:
            self.pooling_mult *= 2
            if self.attention_type == 'factorized':
                position_embeds = position_embeds[:2] + self.stride_pool(position_embeds[2:], 0)
            token_type_mat = self.stride_pool(token_type_mat, 2)
            cls_mask = self.stride_pool(cls_mask, 1)
            attention_mask = self.pool_tensor(attention_mask, mode='min')
        attention_inputs = (position_embeds, token_type_mat, attention_mask, cls_mask)
        return attention_inputs
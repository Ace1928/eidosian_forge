from __future__ import annotations
import collections
import math
from typing import List, Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_layoutlmv3 import LayoutLMv3Config
class TFLayoutLMv3Encoder(keras.layers.Layer):

    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.layer = [TFLayoutLMv3Layer(config, name=f'layer.{i}') for i in range(config.num_hidden_layers)]
        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias
        if self.has_relative_attention_bias:
            self.rel_pos_bins = config.rel_pos_bins
            self.max_rel_pos = config.max_rel_pos
            self.rel_pos_bias = keras.layers.Dense(units=config.num_attention_heads, kernel_initializer=get_initializer(config.initializer_range), use_bias=False, name='rel_pos_bias')
        if self.has_spatial_attention_bias:
            self.max_rel_2d_pos = config.max_rel_2d_pos
            self.rel_2d_pos_bins = config.rel_2d_pos_bins
            self.rel_pos_x_bias = keras.layers.Dense(units=config.num_attention_heads, kernel_initializer=get_initializer(config.initializer_range), use_bias=False, name='rel_pos_x_bias')
            self.rel_pos_y_bias = keras.layers.Dense(units=config.num_attention_heads, kernel_initializer=get_initializer(config.initializer_range), use_bias=False, name='rel_pos_y_bias')

    def relative_position_bucket(self, relative_positions: tf.Tensor, num_buckets: int, max_distance: int):
        num_buckets = num_buckets // 2
        buckets = tf.abs(relative_positions)
        max_exact_buckets = num_buckets // 2
        is_small = buckets < max_exact_buckets
        buckets_log_ratio = tf.math.log(tf.cast(buckets, tf.float32) / max_exact_buckets)
        distance_log_ratio = math.log(max_distance / max_exact_buckets)
        buckets_big_offset = buckets_log_ratio / distance_log_ratio * (num_buckets - max_exact_buckets)
        buckets_big = max_exact_buckets + buckets_big_offset
        buckets_big = tf.cast(buckets_big, buckets.dtype)
        buckets_big = tf.minimum(buckets_big, num_buckets - 1)
        return tf.cast(relative_positions > 0, buckets.dtype) * num_buckets + tf.where(is_small, buckets, buckets_big)

    def _cal_pos_emb(self, dense_layer: keras.layers.Dense, position_ids: tf.Tensor, num_buckets: int, max_distance: int):
        rel_pos_matrix = tf.expand_dims(position_ids, axis=-2) - tf.expand_dims(position_ids, axis=-1)
        rel_pos = self.relative_position_bucket(rel_pos_matrix, num_buckets, max_distance)
        rel_pos_one_hot = tf.one_hot(rel_pos, depth=num_buckets, dtype=self.compute_dtype)
        embedding = dense_layer(rel_pos_one_hot)
        embedding = tf.transpose(embedding, [0, 3, 1, 2])
        embedding = tf.cast(embedding, dtype=self.compute_dtype)
        return embedding

    def _cal_1d_pos_emb(self, position_ids: tf.Tensor):
        return self._cal_pos_emb(self.rel_pos_bias, position_ids, self.rel_pos_bins, self.max_rel_pos)

    def _cal_2d_pos_emb(self, bbox: tf.Tensor):
        position_coord_x = bbox[:, :, 0]
        position_coord_y = bbox[:, :, 3]
        rel_pos_x = self._cal_pos_emb(self.rel_pos_x_bias, position_coord_x, self.rel_2d_pos_bins, self.max_rel_2d_pos)
        rel_pos_y = self._cal_pos_emb(self.rel_pos_y_bias, position_coord_y, self.rel_2d_pos_bins, self.max_rel_2d_pos)
        rel_2d_pos = rel_pos_x + rel_pos_y
        return rel_2d_pos

    def call(self, hidden_states: tf.Tensor, bbox: tf.Tensor | None=None, attention_mask: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True, position_ids: tf.Tensor | None=None, training: bool=False) -> Union[TFBaseModelOutput, Tuple[tf.Tensor], Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        rel_pos = self._cal_1d_pos_emb(position_ids) if self.has_relative_attention_bias else None
        rel_2d_pos = self._cal_2d_pos_emb(bbox) if self.has_spatial_attention_bias else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions, rel_pos=rel_pos, rel_2d_pos=rel_2d_pos, training=training)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if return_dict:
            return TFBaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions)
        else:
            return tuple((value for value in [hidden_states, all_hidden_states, all_self_attentions] if value is not None))

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'rel_pos_bias', None) is not None:
            with tf.name_scope(self.rel_pos_bias.name):
                self.rel_pos_bias.build([None, None, self.rel_pos_bins])
        if getattr(self, 'rel_pos_x_bias', None) is not None:
            with tf.name_scope(self.rel_pos_x_bias.name):
                self.rel_pos_x_bias.build([None, None, self.rel_2d_pos_bins])
        if getattr(self, 'rel_pos_y_bias', None) is not None:
            with tf.name_scope(self.rel_pos_y_bias.name):
                self.rel_pos_y_bias.build([None, None, self.rel_2d_pos_bins])
        if getattr(self, 'layer', None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)
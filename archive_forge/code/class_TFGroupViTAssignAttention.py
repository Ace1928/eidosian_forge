from __future__ import annotations
import collections.abc
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_groupvit import GroupViTConfig, GroupViTTextConfig, GroupViTVisionConfig
class TFGroupViTAssignAttention(keras.layers.Layer):

    def __init__(self, config: GroupViTVisionConfig, **kwargs):
        super().__init__(**kwargs)
        self.scale = config.hidden_size ** (-0.5)
        self.q_proj = keras.layers.Dense(config.hidden_size, name='q_proj')
        self.k_proj = keras.layers.Dense(config.hidden_size, name='k_proj')
        self.v_proj = keras.layers.Dense(config.hidden_size, name='v_proj')
        self.proj = keras.layers.Dense(config.hidden_size, name='proj')
        self.assign_eps = config.assign_eps
        self.config = config

    def get_attn(self, attn: tf.Tensor, gumbel: bool=True, hard: bool=True, training: bool=False) -> tf.Tensor:
        if gumbel and training:
            attn = gumbel_softmax(attn, dim=-2, hard=hard)
        elif hard:
            attn = hard_softmax(attn, dim=-2)
        else:
            attn = stable_softmax(attn, axis=-2)
        return attn

    def call(self, query: tf.Tensor, key: tf.Tensor, training: bool=False):
        value = key
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)
        raw_attn = tf.matmul(query, key, transpose_b=True) * self.scale
        attn = self.get_attn(raw_attn, training=training)
        soft_attn = self.get_attn(raw_attn, training=training, gumbel=False, hard=False)
        attn = attn / (tf.math.reduce_sum(attn, axis=-1, keepdims=True) + self.assign_eps)
        out = tf.matmul(attn, value)
        out = self.proj(out)
        return (out, soft_attn)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'q_proj', None) is not None:
            with tf.name_scope(self.q_proj.name):
                self.q_proj.build([None, None, self.config.hidden_size])
        if getattr(self, 'k_proj', None) is not None:
            with tf.name_scope(self.k_proj.name):
                self.k_proj.build([None, None, self.config.hidden_size])
        if getattr(self, 'v_proj', None) is not None:
            with tf.name_scope(self.v_proj.name):
                self.v_proj.build([None, None, self.config.hidden_size])
        if getattr(self, 'proj', None) is not None:
            with tf.name_scope(self.proj.name):
                self.proj.build([None, None, self.config.hidden_size])
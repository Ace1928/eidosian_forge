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
class TFGroupViTCrossAttentionLayer(keras.layers.Layer):

    def __init__(self, config: GroupViTVisionConfig, **kwargs):
        super().__init__(**kwargs)
        self.attn = TFGroupViTAttention(config, name='attn')
        self.norm2 = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='norm2')
        self.mlp = TFGroupViTMLP(config, name='mlp')
        self.norm_post = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='norm_post')
        self.config = config

    def call(self, query: tf.Tensor, key: tf.Tensor, training: bool=False) -> tf.Tensor:
        x = query
        x = x + self.attn(query, encoder_hidden_states=key)[0]
        x = x + self.mlp(self.norm2(x))
        x = self.norm_post(x)
        return x

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'attn', None) is not None:
            with tf.name_scope(self.attn.name):
                self.attn.build(None)
        if getattr(self, 'norm2', None) is not None:
            with tf.name_scope(self.norm2.name):
                self.norm2.build([None, None, self.config.hidden_size])
        if getattr(self, 'mlp', None) is not None:
            with tf.name_scope(self.mlp.name):
                self.mlp.build(None)
        if getattr(self, 'norm_post', None) is not None:
            with tf.name_scope(self.norm_post.name):
                self.norm_post.build([None, None, self.config.hidden_size])
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
class TFGroupViTTokenAssign(keras.layers.Layer):

    def __init__(self, config: GroupViTVisionConfig, num_group_token: int, num_output_group: int, **kwargs):
        super().__init__(**kwargs)
        self.num_output_group = num_output_group
        self.norm_tokens = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='norm_tokens')
        assign_mlp_ratio = config.assign_mlp_ratio if isinstance(config.assign_mlp_ratio, collections.abc.Iterable) else (config.assign_mlp_ratio, config.assign_mlp_ratio)
        tokens_dim, channels_dim = [int(x * config.hidden_size) for x in assign_mlp_ratio]
        self.mlp_inter = TFGroupViTMixerMLP(config, num_group_token, tokens_dim, num_output_group, name='mlp_inter')
        self.norm_post_tokens = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='norm_post_tokens')
        self.norm_x = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='norm_x')
        self.pre_assign_attn = TFGroupViTCrossAttentionLayer(config, name='pre_assign_attn')
        self.assign = TFGroupViTAssignAttention(config, name='assign')
        self.norm_new_x = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='norm_new_x')
        self.mlp_channels = TFGroupViTMLP(config, config.hidden_size, channels_dim, config.hidden_size, name='mlp_channels')
        self.config = config

    def project_group_token(self, group_tokens: tf.Tensor) -> tf.Tensor:
        """
        Args:
            group_tokens (tf.Tensor): group tokens, [batch_size, num_group_tokens, channels]

        Returns:
            projected_group_tokens (tf.Tensor): [batch_size, num_output_groups, channels]
        """
        projected_group_tokens = self.mlp_inter(group_tokens)
        projected_group_tokens = self.norm_post_tokens(projected_group_tokens)
        return projected_group_tokens

    def call(self, image_tokens: tf.Tensor, group_tokens: tf.Tensor, training: bool=False):
        """
        Args:
            image_tokens (`tf.Tensor`): image tokens, of shape [batch_size, input_length, channels]
            group_tokens (`tf.Tensor`): group tokens, [batch_size, num_group_tokens, channels]
        """
        group_tokens = self.norm_tokens(group_tokens)
        image_tokens = self.norm_x(image_tokens)
        projected_group_tokens = self.project_group_token(group_tokens)
        projected_group_tokens = self.pre_assign_attn(projected_group_tokens, image_tokens)
        new_image_tokens, attention = self.assign(projected_group_tokens, image_tokens)
        new_image_tokens += projected_group_tokens
        new_image_tokens = new_image_tokens + self.mlp_channels(self.norm_new_x(new_image_tokens))
        return (new_image_tokens, attention)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'norm_tokens', None) is not None:
            with tf.name_scope(self.norm_tokens.name):
                self.norm_tokens.build([None, None, self.config.hidden_size])
        if getattr(self, 'mlp_inter', None) is not None:
            with tf.name_scope(self.mlp_inter.name):
                self.mlp_inter.build(None)
        if getattr(self, 'norm_post_tokens', None) is not None:
            with tf.name_scope(self.norm_post_tokens.name):
                self.norm_post_tokens.build([None, None, self.config.hidden_size])
        if getattr(self, 'norm_x', None) is not None:
            with tf.name_scope(self.norm_x.name):
                self.norm_x.build([None, None, self.config.hidden_size])
        if getattr(self, 'pre_assign_attn', None) is not None:
            with tf.name_scope(self.pre_assign_attn.name):
                self.pre_assign_attn.build(None)
        if getattr(self, 'assign', None) is not None:
            with tf.name_scope(self.assign.name):
                self.assign.build(None)
        if getattr(self, 'norm_new_x', None) is not None:
            with tf.name_scope(self.norm_new_x.name):
                self.norm_new_x.build([None, None, self.config.hidden_size])
        if getattr(self, 'mlp_channels', None) is not None:
            with tf.name_scope(self.mlp_channels.name):
                self.mlp_channels.build(None)
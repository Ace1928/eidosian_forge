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
class TFGroupViTStage(keras.layers.Layer):
    """This corresponds to the `GroupingLayer` class in the GroupViT implementation."""

    def __init__(self, config: GroupViTVisionConfig, depth: int, num_prev_group_token: int, num_group_token: int, num_output_group: int, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.depth = depth
        self.num_group_token = num_group_token
        self.layers = [TFGroupViTEncoderLayer(config, name=f'layers_._{i}') for i in range(depth)]
        if num_group_token > 0:
            self.downsample = TFGroupViTTokenAssign(config=config, num_group_token=num_group_token, num_output_group=num_output_group, name='downsample')
        else:
            self.downsample = None
        if num_prev_group_token > 0 and num_group_token > 0:
            self.group_projector = [keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='group_projector.0'), TFGroupViTMixerMLP(config, num_prev_group_token, config.hidden_size // 2, num_group_token, name='group_projector.1')]
        else:
            self.group_projector = None

    def build(self, input_shape=None):
        if self.num_group_token > 0:
            self.group_token = self.add_weight(shape=(1, self.num_group_token, self.config.hidden_size), initializer='zeros', trainable=True, name='group_token')
        else:
            self.group_token = None
        if self.built:
            return
        self.built = True
        if getattr(self, 'downsample', None) is not None:
            with tf.name_scope(self.downsample.name):
                self.downsample.build(None)
        if getattr(self, 'layers', None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)
        if getattr(self, 'group_projector', None) is not None:
            with tf.name_scope(self.group_projector[0].name):
                self.group_projector[0].build([None, None, self.config.hidden_size])
            with tf.name_scope(self.group_projector[1].name):
                self.group_projector[1].build(None)

    @property
    def with_group_token(self):
        return self.group_token is not None

    def split_x(self, x: tf.Tensor) -> tf.Tensor:
        if self.with_group_token:
            return (x[:, :-self.num_group_token], x[:, -self.num_group_token:])
        else:
            return (x, None)

    def concat_x(self, x: tf.Tensor, group_token: tf.Tensor | None=None) -> tf.Tensor:
        if group_token is None:
            return x
        return tf.concat([x, group_token], axis=1)

    def call(self, hidden_states: tf.Tensor, prev_group_token: tf.Tensor | None=None, output_attentions: bool=False, training: bool=False) -> Tuple[tf.Tensor]:
        """
        Args:
            hidden_states (`tf.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`tf.Tensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the grouping tensors of Grouping block.
        """
        if self.with_group_token:
            group_token = tf.tile(self.group_token, multiples=(shape_list(hidden_states)[0], 1, 1))
            if self.group_projector is not None:
                for layer in self.group_projector:
                    prev_group_token = layer(prev_group_token)
                group_token = group_token + prev_group_token
        else:
            group_token = None
        x = hidden_states
        cat_x = self.concat_x(x, group_token)
        for layer in self.layers:
            layer_out = layer(cat_x, attention_mask=None, causal_attention_mask=None, output_attentions=None)
            cat_x = layer_out[0]
        x, group_token = self.split_x(cat_x)
        attention = None
        if self.downsample is not None:
            x, attention = self.downsample(x, group_token)
        outputs = (x, group_token)
        if output_attentions:
            outputs = outputs + (attention,)
        return outputs
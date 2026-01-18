from __future__ import annotations
import collections.abc
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import tensorflow as tf
from ...modeling_tf_outputs import TFImageClassifierOutputWithNoAttention
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_cvt import CvtConfig
class TFCvtStage(keras.layers.Layer):
    """
    Cvt stage (encoder block). Each stage has 2 parts :
    - (1) A Convolutional Token Embedding layer
    - (2) A Convolutional Transformer Block (layer).
    The classification token is added only in the last stage.

    Args:
        config ([`CvtConfig`]): Model configuration class.
        stage (`int`): Stage number.
    """

    def __init__(self, config: CvtConfig, stage: int, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.stage = stage
        if self.config.cls_token[self.stage]:
            self.cls_token = self.add_weight(shape=(1, 1, self.config.embed_dim[-1]), initializer=get_initializer(self.config.initializer_range), trainable=True, name='cvt.encoder.stages.2.cls_token')
        self.embedding = TFCvtEmbeddings(self.config, patch_size=config.patch_sizes[self.stage], num_channels=config.num_channels if self.stage == 0 else config.embed_dim[self.stage - 1], stride=config.patch_stride[self.stage], embed_dim=config.embed_dim[self.stage], padding=config.patch_padding[self.stage], dropout_rate=config.drop_rate[self.stage], name='embedding')
        drop_path_rates = tf.linspace(0.0, config.drop_path_rate[self.stage], config.depth[stage])
        drop_path_rates = [x.numpy().item() for x in drop_path_rates]
        self.layers = [TFCvtLayer(config, num_heads=config.num_heads[self.stage], embed_dim=config.embed_dim[self.stage], kernel_size=config.kernel_qkv[self.stage], stride_q=config.stride_q[self.stage], stride_kv=config.stride_kv[self.stage], padding_q=config.padding_q[self.stage], padding_kv=config.padding_kv[self.stage], qkv_projection_method=config.qkv_projection_method[self.stage], qkv_bias=config.qkv_bias[self.stage], attention_drop_rate=config.attention_drop_rate[self.stage], drop_rate=config.drop_rate[self.stage], mlp_ratio=config.mlp_ratio[self.stage], drop_path_rate=drop_path_rates[self.stage], with_cls_token=config.cls_token[self.stage], name=f'layers.{j}') for j in range(config.depth[self.stage])]

    def call(self, hidden_state: tf.Tensor, training: bool=False):
        cls_token = None
        hidden_state = self.embedding(hidden_state, training)
        batch_size, height, width, num_channels = shape_list(hidden_state)
        hidden_size = height * width
        hidden_state = tf.reshape(hidden_state, shape=(batch_size, hidden_size, num_channels))
        if self.config.cls_token[self.stage]:
            cls_token = tf.repeat(self.cls_token, repeats=batch_size, axis=0)
            hidden_state = tf.concat((cls_token, hidden_state), axis=1)
        for layer in self.layers:
            layer_outputs = layer(hidden_state, height, width, training=training)
            hidden_state = layer_outputs
        if self.config.cls_token[self.stage]:
            cls_token, hidden_state = tf.split(hidden_state, [1, height * width], 1)
        hidden_state = tf.reshape(hidden_state, shape=(batch_size, height, width, num_channels))
        return (hidden_state, cls_token)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'embedding', None) is not None:
            with tf.name_scope(self.embedding.name):
                self.embedding.build(None)
        if getattr(self, 'layers', None) is not None:
            for layer in self.layers:
                with tf.name_scope(layer.name):
                    layer.build(None)
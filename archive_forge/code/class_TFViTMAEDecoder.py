from __future__ import annotations
import collections.abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...file_utils import (
from ...modeling_tf_outputs import TFBaseModelOutput
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import logging
from .configuration_vit_mae import ViTMAEConfig
class TFViTMAEDecoder(keras.layers.Layer):

    def __init__(self, config, num_patches, **kwargs):
        super().__init__(**kwargs)
        self.decoder_embed = keras.layers.Dense(config.decoder_hidden_size, name='decoder_embed')
        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        self.decoder_layers = [TFViTMAELayer(decoder_config, name=f'decoder_layers.{j}') for j in range(config.decoder_num_hidden_layers)]
        self.decoder_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='decoder_norm')
        self.decoder_pred = keras.layers.Dense(config.patch_size ** 2 * config.num_channels, kernel_initializer=get_initializer(config.initializer_range), name='decoder_pred')
        self.config = config
        self.num_patches = num_patches

    def build(self, input_shape=None):
        self.mask_token = self.add_weight(shape=(1, 1, self.config.decoder_hidden_size), initializer=tf.random_normal_initializer(stddev=self.config.initializer_range), trainable=True, name='mask_token')
        self.decoder_pos_embed = self.add_weight(shape=(1, self.num_patches + 1, self.config.decoder_hidden_size), initializer='zeros', trainable=False, name='decoder_pos_embed')
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches ** 0.5), add_cls_token=True)[None, ...]
        self.decoder_pos_embed.assign(decoder_pos_embed)
        if self.built:
            return
        self.built = True
        if getattr(self, 'decoder_embed', None) is not None:
            with tf.name_scope(self.decoder_embed.name):
                self.decoder_embed.build([None, None, self.config.hidden_size])
        if getattr(self, 'decoder_norm', None) is not None:
            with tf.name_scope(self.decoder_norm.name):
                self.decoder_norm.build([None, None, self.config.decoder_hidden_size])
        if getattr(self, 'decoder_pred', None) is not None:
            with tf.name_scope(self.decoder_pred.name):
                self.decoder_pred.build([None, None, self.config.decoder_hidden_size])
        if getattr(self, 'decoder_layers', None) is not None:
            for layer in self.decoder_layers:
                with tf.name_scope(layer.name):
                    layer.build(None)

    def call(self, hidden_states, ids_restore, output_attentions=False, output_hidden_states=False, return_dict=True):
        x = self.decoder_embed(hidden_states)
        mask_tokens = tf.tile(self.mask_token, (shape_list(x)[0], shape_list(ids_restore)[1] + 1 - shape_list(x)[1], 1))
        x_ = tf.concat([x[:, 1:, :], mask_tokens], axis=1)
        x_ = tf.gather(x_, axis=1, batch_dims=1, indices=ids_restore)
        x = tf.concat([x[:, :1, :], x_], axis=1)
        hidden_states = x + self.decoder_pos_embed
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.decoder_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states, head_mask=None, output_attentions=output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        hidden_states = self.decoder_norm(hidden_states)
        logits = self.decoder_pred(hidden_states)
        logits = logits[:, 1:, :]
        if not return_dict:
            return tuple((v for v in [logits, all_hidden_states, all_self_attentions] if v is not None))
        return TFViTMAEDecoderOutput(logits=logits, hidden_states=all_hidden_states, attentions=all_self_attentions)
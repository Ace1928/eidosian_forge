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
class TFGroupViTTextTransformer(keras.layers.Layer):

    def __init__(self, config: GroupViTTextConfig, **kwargs):
        super().__init__(**kwargs)
        self.embeddings = TFGroupViTTextEmbeddings(config, name='embeddings')
        self.encoder = TFGroupViTTextEncoder(config, name='encoder')
        self.final_layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='final_layer_norm')
        self.eos_token_id = config.eos_token_id
        self.embed_dim = config.hidden_size

    def call(self, input_ids: TFModelInputType, attention_mask: tf.Tensor, position_ids: tf.Tensor, output_attentions: bool, output_hidden_states: bool, return_dict: bool, training: bool=False) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        input_shape = shape_list(input_ids)
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids)
        batch_size, seq_length = input_shape
        causal_attention_mask = self._build_causal_attention_mask(batch_size, seq_length, dtype=embedding_output.dtype)
        attention_mask = _expand_mask(attention_mask)
        encoder_outputs = self.encoder(hidden_states=embedding_output, attention_mask=attention_mask, causal_attention_mask=causal_attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = encoder_outputs[0]
        sequence_output = self.final_layer_norm(inputs=sequence_output)
        if self.eos_token_id == 2:
            pooled_output = tf.gather_nd(params=sequence_output, indices=tf.stack(values=(tf.range(input_shape[0], dtype=tf.int64), tf.math.argmax(input_ids, axis=-1)), axis=1))
        else:
            pooled_output = tf.gather_nd(params=sequence_output, indices=tf.stack(values=(tf.range(input_shape[0], dtype=tf.int64), tf.math.argmax(tf.cast(input_ids == self.eos_token_id, dtype=tf.int8), axis=-1)), axis=1))
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        return TFBaseModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

    def _build_causal_attention_mask(self, batch_size, seq_length, dtype=tf.float32):
        diag = tf.cast(tf.fill((seq_length,), 0.0), dtype)
        to_mask = tf.cast(tf.fill((seq_length, seq_length), -10000.0), dtype)
        to_mask = tf.linalg.band_part(to_mask, 0, -1)
        to_mask = tf.linalg.set_diag(to_mask, diagonal=diag)
        return tf.broadcast_to(input=to_mask, shape=(batch_size, 1, seq_length, seq_length))

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'embeddings', None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        if getattr(self, 'encoder', None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        if getattr(self, 'final_layer_norm', None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build([None, None, self.embed_dim])
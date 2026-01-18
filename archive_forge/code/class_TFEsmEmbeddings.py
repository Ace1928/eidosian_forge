from __future__ import annotations
import os
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, stable_softmax
from ...utils import logging
from .configuration_esm import EsmConfig
class TFEsmEmbeddings(keras.layers.Layer):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.word_embeddings = keras.layers.Embedding(config.vocab_size, config.hidden_size, embeddings_initializer=get_initializer(config.initializer_range), name='word_embeddings')
        self.position_embeddings = keras.layers.Embedding(config.max_position_embeddings, config.hidden_size, embeddings_initializer=get_initializer(config.initializer_range), name='position_embeddings')
        if config.emb_layer_norm_before:
            self.layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layer_norm')
        else:
            self.layer_norm = None
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'absolute')
        self.position_ids = tf.range(config.max_position_embeddings)[None, :]
        self.padding_idx = config.pad_token_id
        self.token_dropout = config.token_dropout
        self.mask_token_id = config.mask_token_id
        self.config = config

    def call(self, input_ids=None, attention_mask=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        if position_ids is None:
            if input_ids is not None:
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)
        if inputs_embeds is None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = self.word_embeddings(input_ids)
        embeddings = inputs_embeds
        if self.token_dropout:
            embeddings = tf.where((input_ids == self.mask_token_id)[:, :, None], 0.0, embeddings)
            mask_ratio_train = 0.15 * 0.8
            src_lengths = tf.cast(tf.reduce_sum(attention_mask, axis=-1), tf.float32)
            masked_tokens = input_ids == self.mask_token_id
            mask_ratio_observed = tf.math.count_nonzero(masked_tokens, dtype=tf.float32, axis=-1) / src_lengths
            embeddings = embeddings * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]
        if self.position_embedding_type == 'absolute':
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        if self.layer_norm is not None:
            embeddings = self.layer_norm(embeddings)
        if attention_mask is not None:
            embeddings = embeddings * tf.cast(tf.expand_dims(attention_mask, -1), embeddings.dtype)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: tf.Tensor

        Returns: tf.Tensor
        """
        input_shape = shape_list(inputs_embeds)[:-1]
        sequence_length = input_shape[1]
        position_ids = tf.range(start=self.padding_idx + 1, limit=sequence_length + self.padding_idx + 1, dtype=tf.int64)
        return tf.broadcast_to(tf.expand_dims(position_ids, 0), input_shape)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'word_embeddings', None) is not None:
            with tf.name_scope(self.word_embeddings.name):
                self.word_embeddings.build(None)
        if getattr(self, 'position_embeddings', None) is not None:
            with tf.name_scope(self.position_embeddings.name):
                self.position_embeddings.build(None)
        if getattr(self, 'layer_norm', None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.hidden_size])
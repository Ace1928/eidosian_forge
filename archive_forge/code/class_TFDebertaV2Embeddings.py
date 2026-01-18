from __future__ import annotations
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_deberta_v2 import DebertaV2Config
class TFDebertaV2Embeddings(keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.embedding_size = getattr(config, 'embedding_size', config.hidden_size)
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.position_biased_input = getattr(config, 'position_biased_input', True)
        self.initializer_range = config.initializer_range
        if self.embedding_size != config.hidden_size:
            self.embed_proj = keras.layers.Dense(config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name='embed_proj', use_bias=False)
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='LayerNorm')
        self.dropout = TFDebertaV2StableDropout(config.hidden_dropout_prob, name='dropout')

    def build(self, input_shape=None):
        with tf.name_scope('word_embeddings'):
            self.weight = self.add_weight(name='weight', shape=[self.config.vocab_size, self.embedding_size], initializer=get_initializer(self.initializer_range))
        with tf.name_scope('token_type_embeddings'):
            if self.config.type_vocab_size > 0:
                self.token_type_embeddings = self.add_weight(name='embeddings', shape=[self.config.type_vocab_size, self.embedding_size], initializer=get_initializer(self.initializer_range))
            else:
                self.token_type_embeddings = None
        with tf.name_scope('position_embeddings'):
            if self.position_biased_input:
                self.position_embeddings = self.add_weight(name='embeddings', shape=[self.max_position_embeddings, self.hidden_size], initializer=get_initializer(self.initializer_range))
            else:
                self.position_embeddings = None
        if self.built:
            return
        self.built = True
        if getattr(self, 'LayerNorm', None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
        if getattr(self, 'dropout', None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)
        if getattr(self, 'embed_proj', None) is not None:
            with tf.name_scope(self.embed_proj.name):
                self.embed_proj.build([None, None, self.embedding_size])

    def call(self, input_ids: tf.Tensor=None, position_ids: tf.Tensor=None, token_type_ids: tf.Tensor=None, inputs_embeds: tf.Tensor=None, mask: tf.Tensor=None, training: bool=False) -> tf.Tensor:
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        if input_ids is None and inputs_embeds is None:
            raise ValueError('Need to provide either `input_ids` or `input_embeds`.')
        if input_ids is not None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)
        input_shape = shape_list(inputs_embeds)[:-1]
        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)
        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)
        final_embeddings = inputs_embeds
        if self.position_biased_input:
            position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
            final_embeddings += position_embeds
        if self.config.type_vocab_size > 0:
            token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)
            final_embeddings += token_type_embeds
        if self.embedding_size != self.hidden_size:
            final_embeddings = self.embed_proj(final_embeddings)
        final_embeddings = self.LayerNorm(final_embeddings)
        if mask is not None:
            if len(shape_list(mask)) != len(shape_list(final_embeddings)):
                if len(shape_list(mask)) == 4:
                    mask = tf.squeeze(tf.squeeze(mask, axis=1), axis=1)
                mask = tf.cast(tf.expand_dims(mask, axis=2), tf.float32)
            final_embeddings = final_embeddings * mask
        final_embeddings = self.dropout(final_embeddings, training=training)
        return final_embeddings
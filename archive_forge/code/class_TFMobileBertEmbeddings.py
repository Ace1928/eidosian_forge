from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_mobilebert import MobileBertConfig
class TFMobileBertEmbeddings(keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.trigram_input = config.trigram_input
        self.embedding_size = config.embedding_size
        self.config = config
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.initializer_range = config.initializer_range
        self.embedding_transformation = keras.layers.Dense(config.hidden_size, name='embedding_transformation')
        self.LayerNorm = NORM2FN[config.normalization_type](config.hidden_size, epsilon=config.layer_norm_eps, name='LayerNorm')
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)
        self.embedded_input_size = self.embedding_size * (3 if self.trigram_input else 1)

    def build(self, input_shape=None):
        with tf.name_scope('word_embeddings'):
            self.weight = self.add_weight(name='weight', shape=[self.config.vocab_size, self.embedding_size], initializer=get_initializer(initializer_range=self.initializer_range))
        with tf.name_scope('token_type_embeddings'):
            self.token_type_embeddings = self.add_weight(name='embeddings', shape=[self.config.type_vocab_size, self.hidden_size], initializer=get_initializer(initializer_range=self.initializer_range))
        with tf.name_scope('position_embeddings'):
            self.position_embeddings = self.add_weight(name='embeddings', shape=[self.max_position_embeddings, self.hidden_size], initializer=get_initializer(initializer_range=self.initializer_range))
        if self.built:
            return
        self.built = True
        if getattr(self, 'embedding_transformation', None) is not None:
            with tf.name_scope(self.embedding_transformation.name):
                self.embedding_transformation.build([None, None, self.embedded_input_size])
        if getattr(self, 'LayerNorm', None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build(None)

    def call(self, input_ids=None, position_ids=None, token_type_ids=None, inputs_embeds=None, training=False):
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        assert not (input_ids is None and inputs_embeds is None)
        if input_ids is not None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)
        input_shape = shape_list(inputs_embeds)[:-1]
        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)
        if self.trigram_input:
            inputs_embeds = tf.concat([tf.pad(inputs_embeds[:, 1:], ((0, 0), (0, 1), (0, 0))), inputs_embeds, tf.pad(inputs_embeds[:, :-1], ((0, 0), (1, 0), (0, 0)))], axis=2)
        if self.trigram_input or self.embedding_size != self.hidden_size:
            inputs_embeds = self.embedding_transformation(inputs_embeds)
        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)
        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)
        final_embeddings = inputs_embeds + position_embeds + token_type_embeds
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)
        return final_embeddings
from __future__ import annotations
import enum
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_tapas import TapasConfig
class TFTapasEmbeddings(keras.layers.Layer):
    """
    Construct the embeddings from word, position and token_type embeddings. Same as BertEmbeddings but with a number of
    additional token type embeddings to encode tabular structure.
    """

    def __init__(self, config: TapasConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.number_of_token_type_embeddings = len(config.type_vocab_sizes)
        self.reset_position_index_per_cell = config.reset_position_index_per_cell
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.initializer_range = config.initializer_range
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='LayerNorm')
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def build(self, input_shape=None):
        with tf.name_scope('word_embeddings'):
            self.weight = self.add_weight(name='weight', shape=[self.config.vocab_size, self.hidden_size], initializer=get_initializer(self.initializer_range))
        with tf.name_scope('position_embeddings'):
            self.position_embeddings = self.add_weight(name='embeddings', shape=[self.max_position_embeddings, self.hidden_size], initializer=get_initializer(self.initializer_range))
        for i, type_vocab_size in enumerate(self.config.type_vocab_sizes):
            with tf.name_scope(f'token_type_embeddings_{i}'):
                setattr(self, f'token_type_embeddings_{i}', self.add_weight(name='embeddings', shape=[type_vocab_size, self.hidden_size], initializer=get_initializer(self.initializer_range)))
        if self.built:
            return
        self.built = True
        if getattr(self, 'LayerNorm', None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])

    def call(self, input_ids: tf.Tensor=None, position_ids: tf.Tensor=None, token_type_ids: tf.Tensor=None, inputs_embeds: tf.Tensor=None, training: bool=False) -> tf.Tensor:
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        assert not (input_ids is None and inputs_embeds is None)
        if input_ids is not None:
            input_shape = shape_list(input_ids)
        else:
            input_shape = shape_list(inputs_embeds)[:-1]
        seq_length = input_shape[1]
        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape + [self.number_of_token_type_embeddings], value=0)
        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(start=0, limit=seq_length), axis=0)
            position_ids = tf.broadcast_to(position_ids, shape=input_shape)
            if self.reset_position_index_per_cell:
                col_index = IndexMap(token_type_ids[:, :, 1], self.config.type_vocab_sizes[1], batch_dims=1)
                row_index = IndexMap(token_type_ids[:, :, 2], self.config.type_vocab_sizes[2], batch_dims=1)
                full_index = ProductIndexMap(col_index, row_index)
                first_position_per_segment = reduce_min(position_ids, full_index)[0]
                first_position = gather(first_position_per_segment, full_index)
                position = tf.expand_dims(tf.range(start=0, limit=seq_length), axis=0)
                position_ids = tf.math.minimum(self.max_position_embeddings - 1, position - first_position)
        if input_ids is not None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)
        position_embeddings = tf.gather(self.position_embeddings, indices=position_ids)
        final_embeddings = inputs_embeds + position_embeddings
        for i in range(self.number_of_token_type_embeddings):
            name = f'token_type_embeddings_{i}'
            final_embeddings += tf.gather(params=getattr(self, name), indices=token_type_ids[:, :, i])
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)
        return final_embeddings
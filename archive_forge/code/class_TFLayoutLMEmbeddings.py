from __future__ import annotations
import math
import warnings
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_layoutlm import LayoutLMConfig
class TFLayoutLMEmbeddings(keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config: LayoutLMConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.max_2d_position_embeddings = config.max_2d_position_embeddings
        self.initializer_range = config.initializer_range
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='LayerNorm')
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def build(self, input_shape=None):
        with tf.name_scope('word_embeddings'):
            self.weight = self.add_weight(name='weight', shape=[self.config.vocab_size, self.hidden_size], initializer=get_initializer(self.initializer_range))
        with tf.name_scope('token_type_embeddings'):
            self.token_type_embeddings = self.add_weight(name='embeddings', shape=[self.config.type_vocab_size, self.hidden_size], initializer=get_initializer(self.initializer_range))
        with tf.name_scope('position_embeddings'):
            self.position_embeddings = self.add_weight(name='embeddings', shape=[self.max_position_embeddings, self.hidden_size], initializer=get_initializer(self.initializer_range))
        with tf.name_scope('x_position_embeddings'):
            self.x_position_embeddings = self.add_weight(name='embeddings', shape=[self.max_2d_position_embeddings, self.hidden_size], initializer=get_initializer(self.initializer_range))
        with tf.name_scope('y_position_embeddings'):
            self.y_position_embeddings = self.add_weight(name='embeddings', shape=[self.max_2d_position_embeddings, self.hidden_size], initializer=get_initializer(self.initializer_range))
        with tf.name_scope('h_position_embeddings'):
            self.h_position_embeddings = self.add_weight(name='embeddings', shape=[self.max_2d_position_embeddings, self.hidden_size], initializer=get_initializer(self.initializer_range))
        with tf.name_scope('w_position_embeddings'):
            self.w_position_embeddings = self.add_weight(name='embeddings', shape=[self.max_2d_position_embeddings, self.hidden_size], initializer=get_initializer(self.initializer_range))
        if self.built:
            return
        self.built = True
        if getattr(self, 'LayerNorm', None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])

    def call(self, input_ids: tf.Tensor=None, bbox: tf.Tensor=None, position_ids: tf.Tensor=None, token_type_ids: tf.Tensor=None, inputs_embeds: tf.Tensor=None, training: bool=False) -> tf.Tensor:
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
        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)
        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)
        if bbox is None:
            bbox = bbox = tf.fill(input_shape + [4], value=0)
        try:
            left_position_embeddings = tf.gather(self.x_position_embeddings, bbox[:, :, 0])
            upper_position_embeddings = tf.gather(self.y_position_embeddings, bbox[:, :, 1])
            right_position_embeddings = tf.gather(self.x_position_embeddings, bbox[:, :, 2])
            lower_position_embeddings = tf.gather(self.y_position_embeddings, bbox[:, :, 3])
        except IndexError as e:
            raise IndexError('The `bbox`coordinate values should be within 0-1000 range.') from e
        h_position_embeddings = tf.gather(self.h_position_embeddings, bbox[:, :, 3] - bbox[:, :, 1])
        w_position_embeddings = tf.gather(self.w_position_embeddings, bbox[:, :, 2] - bbox[:, :, 0])
        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)
        final_embeddings = inputs_embeds + position_embeds + token_type_embeds + left_position_embeddings + upper_position_embeddings + right_position_embeddings + lower_position_embeddings + h_position_embeddings + w_position_embeddings
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)
        return final_embeddings
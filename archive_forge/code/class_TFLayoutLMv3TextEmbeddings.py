from __future__ import annotations
import collections
import math
from typing import List, Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_layoutlmv3 import LayoutLMv3Config
class TFLayoutLMv3TextEmbeddings(keras.layers.Layer):
    """
    LayoutLMv3 text embeddings. Same as `RobertaEmbeddings` but with added spatial (layout) embeddings.
    """

    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(**kwargs)
        self.word_embeddings = keras.layers.Embedding(config.vocab_size, config.hidden_size, embeddings_initializer=get_initializer(config.initializer_range), name='word_embeddings')
        self.token_type_embeddings = keras.layers.Embedding(config.type_vocab_size, config.hidden_size, embeddings_initializer=get_initializer(config.initializer_range), name='token_type_embeddings')
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='LayerNorm')
        self.dropout = keras.layers.Dropout(config.hidden_dropout_prob)
        self.padding_token_index = config.pad_token_id
        self.position_embeddings = keras.layers.Embedding(config.max_position_embeddings, config.hidden_size, embeddings_initializer=get_initializer(config.initializer_range), name='position_embeddings')
        self.x_position_embeddings = keras.layers.Embedding(config.max_2d_position_embeddings, config.coordinate_size, embeddings_initializer=get_initializer(config.initializer_range), name='x_position_embeddings')
        self.y_position_embeddings = keras.layers.Embedding(config.max_2d_position_embeddings, config.coordinate_size, embeddings_initializer=get_initializer(config.initializer_range), name='y_position_embeddings')
        self.h_position_embeddings = keras.layers.Embedding(config.max_2d_position_embeddings, config.shape_size, embeddings_initializer=get_initializer(config.initializer_range), name='h_position_embeddings')
        self.w_position_embeddings = keras.layers.Embedding(config.max_2d_position_embeddings, config.shape_size, embeddings_initializer=get_initializer(config.initializer_range), name='w_position_embeddings')
        self.max_2d_positions = config.max_2d_position_embeddings
        self.config = config

    def calculate_spatial_position_embeddings(self, bbox: tf.Tensor) -> tf.Tensor:
        try:
            left_position_ids = bbox[:, :, 0]
            upper_position_ids = bbox[:, :, 1]
            right_position_ids = bbox[:, :, 2]
            lower_position_ids = bbox[:, :, 3]
        except IndexError as exception:
            raise IndexError('Bounding box is not of shape (batch_size, seq_length, 4).') from exception
        try:
            left_position_embeddings = self.x_position_embeddings(left_position_ids)
            upper_position_embeddings = self.y_position_embeddings(upper_position_ids)
            right_position_embeddings = self.x_position_embeddings(right_position_ids)
            lower_position_embeddings = self.y_position_embeddings(lower_position_ids)
        except IndexError as exception:
            raise IndexError(f'The `bbox` coordinate values should be within 0-{self.max_2d_positions} range.') from exception
        max_position_id = self.max_2d_positions - 1
        h_position_embeddings = self.h_position_embeddings(tf.clip_by_value(bbox[:, :, 3] - bbox[:, :, 1], 0, max_position_id))
        w_position_embeddings = self.w_position_embeddings(tf.clip_by_value(bbox[:, :, 2] - bbox[:, :, 0], 0, max_position_id))
        spatial_position_embeddings = tf.concat([left_position_embeddings, upper_position_embeddings, right_position_embeddings, lower_position_embeddings, h_position_embeddings, w_position_embeddings], axis=-1)
        return spatial_position_embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embds: tf.Tensor) -> tf.Tensor:
        """
        We are provided embeddings directly. We cannot infer which are padded, so just generate sequential position
        ids.
        """
        input_shape = tf.shape(inputs_embds)
        sequence_length = input_shape[1]
        start_index = self.padding_token_index + 1
        end_index = self.padding_token_index + sequence_length + 1
        position_ids = tf.range(start_index, end_index, dtype=tf.int32)
        batch_size = input_shape[0]
        position_ids = tf.reshape(position_ids, (1, sequence_length))
        position_ids = tf.tile(position_ids, (batch_size, 1))
        return position_ids

    def create_position_ids_from_input_ids(self, input_ids: tf.Tensor) -> tf.Tensor:
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_token_index + 1.
        """
        mask = tf.cast(tf.not_equal(input_ids, self.padding_token_index), input_ids.dtype)
        position_ids = tf.cumsum(mask, axis=1) * mask
        position_ids = position_ids + self.padding_token_index
        return position_ids

    def create_position_ids(self, input_ids: tf.Tensor, inputs_embeds: tf.Tensor) -> tf.Tensor:
        if input_ids is None:
            return self.create_position_ids_from_inputs_embeds(inputs_embeds)
        else:
            return self.create_position_ids_from_input_ids(input_ids)

    def call(self, input_ids: tf.Tensor | None=None, bbox: tf.Tensor=None, token_type_ids: tf.Tensor | None=None, position_ids: tf.Tensor | None=None, inputs_embeds: tf.Tensor | None=None, training: bool=False) -> tf.Tensor:
        if position_ids is None:
            position_ids = self.create_position_ids(input_ids, inputs_embeds)
        if input_ids is not None:
            input_shape = tf.shape(input_ids)
        else:
            input_shape = tf.shape(inputs_embeds)[:-1]
        if token_type_ids is None:
            token_type_ids = tf.zeros(input_shape, dtype=position_ids.dtype)
        if inputs_embeds is None:
            check_embeddings_within_bounds(input_ids, self.word_embeddings.input_dim)
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings
        spatial_position_embeddings = self.calculate_spatial_position_embeddings(bbox)
        embeddings += spatial_position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings, training=training)
        return embeddings

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'word_embeddings', None) is not None:
            with tf.name_scope(self.word_embeddings.name):
                self.word_embeddings.build(None)
        if getattr(self, 'token_type_embeddings', None) is not None:
            with tf.name_scope(self.token_type_embeddings.name):
                self.token_type_embeddings.build(None)
        if getattr(self, 'LayerNorm', None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
        if getattr(self, 'position_embeddings', None) is not None:
            with tf.name_scope(self.position_embeddings.name):
                self.position_embeddings.build(None)
        if getattr(self, 'x_position_embeddings', None) is not None:
            with tf.name_scope(self.x_position_embeddings.name):
                self.x_position_embeddings.build(None)
        if getattr(self, 'y_position_embeddings', None) is not None:
            with tf.name_scope(self.y_position_embeddings.name):
                self.y_position_embeddings.build(None)
        if getattr(self, 'h_position_embeddings', None) is not None:
            with tf.name_scope(self.h_position_embeddings.name):
                self.h_position_embeddings.build(None)
        if getattr(self, 'w_position_embeddings', None) is not None:
            with tf.name_scope(self.w_position_embeddings.name):
                self.w_position_embeddings.build(None)
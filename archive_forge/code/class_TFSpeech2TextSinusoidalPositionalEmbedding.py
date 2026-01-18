from __future__ import annotations
import random
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation, glu
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_speech_to_text import Speech2TextConfig
class TFSpeech2TextSinusoidalPositionalEmbedding(keras.layers.Layer):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int]=None, **kwargs):
        super().__init__(**kwargs)
        self.offset = 2
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.embedding_weights = self._get_embedding(num_positions + self.offset, embedding_dim, padding_idx)

    @staticmethod
    def _get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int]=None) -> tf.Tensor:
        """
        Build sinusoidal embeddings. This matches the implementation in tensor2tensor, but differs slightly from the
        description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = tf.math.log(10000.0) / (half_dim - 1)
        emb = tf.math.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
        emb = tf.expand_dims(tf.range(num_embeddings, dtype=tf.float32), axis=1) * tf.expand_dims(emb, axis=0)
        emb = tf.reshape(tf.concat([tf.math.sin(emb), tf.math.cos(emb)], axis=1), shape=[num_embeddings, -1])
        if embedding_dim % 2 == 1:
            emb = tf.concat([emb, tf.zeros(num_embeddings, 1)], axis=1)
        if padding_idx is not None:
            emb = tf.concat([emb[:padding_idx, :], tf.zeros((1, tf.shape(emb)[1])), emb[padding_idx + 1:, :]], axis=0)
        return emb

    def call(self, input_ids: tf.Tensor, past_key_values_length: int=0) -> tf.Tensor:
        bsz, seq_len = shape_list(input_ids)
        position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
        embeddings = self._get_embedding(self.padding_idx + 1 + seq_len + self.offset + past_key_values_length, self.embedding_dim, self.padding_idx)
        return tf.reshape(tf.gather(embeddings, tf.reshape(position_ids, (-1,)), axis=0), (bsz, seq_len, -1))

    @staticmethod
    def create_position_ids_from_input_ids(input_ids: tf.Tensor, padding_idx: int, past_key_values_length: Optional[int]=0) -> tf.Tensor:
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
        symbols are ignored. This is modified from fairseq's `utils.make_positions`.

        Args:
            x: tf.Tensor x:
        Returns: tf.Tensor
        """
        mask = tf.cast(tf.math.not_equal(input_ids, padding_idx), dtype=tf.int32)
        incremental_indices = (tf.math.cumsum(mask, axis=1) + past_key_values_length) * mask
        return tf.cast(incremental_indices, dtype=tf.int64) + padding_idx
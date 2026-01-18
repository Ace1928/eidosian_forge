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
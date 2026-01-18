import collections
import math
import string
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import constraints
from keras.src import initializers
from keras.src import regularizers
from keras.src.engine.base_layer import Layer
from keras.src.layers import activation
from keras.src.layers import core
from keras.src.layers import regularization
from keras.src.utils import tf_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
def _compute_attention_mask(self, query, value, key=None, attention_mask=None, use_causal_mask=False):
    """Computes the attention mask, using the Keras masks of the inputs.

        * The `query`'s mask is reshaped from [B, T] to [B, T, 1].
        * The `value`'s mask is reshaped from [B, S] to [B, 1, S].
        * The `key`'s mask is reshaped from [B, S] to [B, 1, S]. The `key`'s
          mask is ignored if `key` is `None` or if `key is value`.
        * If `use_causal_mask=True`, then the causal mask is computed. Its shape
          is [1, T, S].

        All defined masks are merged using a logical AND operation (`&`).

        In general, if the `query` and `value` are masked, then there is no need
        to define the `attention_mask`.

        Args:
            query: Projected query `Tensor` of shape `(B, T, N, key_dim)`.
            key: Projected key `Tensor` of shape `(B, T, N, key_dim)`.
            value: Projected value `Tensor` of shape `(B, T, N, value_dim)`.
            attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
                attention to certain positions.
            use_causal_mask: A boolean to indicate whether to apply a causal
                mask to prevent tokens from attending to future tokens (e.g.,
                used in a decoder Transformer).

        Returns:
            attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
                attention to certain positions, based on the Keras masks of the
                `query`, `key`, `value`, and `attention_mask` tensors, and the
                causal mask if `use_causal_mask=True`.
        """
    query_mask = getattr(query, '_keras_mask', None)
    value_mask = getattr(value, '_keras_mask', None)
    key_mask = getattr(key, '_keras_mask', None)
    auto_mask = None
    if query_mask is not None:
        query_mask = tf.cast(query_mask, tf.bool)
        auto_mask = query_mask[:, :, tf.newaxis]
    if value_mask is not None:
        value_mask = tf.cast(value_mask, tf.bool)
        mask = value_mask[:, tf.newaxis, :]
        auto_mask = mask if auto_mask is None else auto_mask & mask
    if key_mask is not None:
        key_mask = tf.cast(key_mask, tf.bool)
        mask = key_mask[:, tf.newaxis, :]
        auto_mask = mask if auto_mask is None else auto_mask & mask
    if use_causal_mask:
        mask = self._compute_causal_mask(query, value)
        auto_mask = mask if auto_mask is None else auto_mask & mask
    if auto_mask is not None:
        attention_mask = auto_mask if attention_mask is None else tf.cast(attention_mask, bool) & auto_mask
    return attention_mask
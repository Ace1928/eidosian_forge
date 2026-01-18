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
def _compute_attention(self, query, key, value, attention_mask=None, training=None):
    """Applies Dot-product attention with query, key, value tensors.

        This function defines the computation inside `call` with projected
        multi-head Q, K, V inputs. Users can override this function for
        customized attention implementation.

        Args:
            query: Projected query `Tensor` of shape `(B, T, N, key_dim)`.
            key: Projected key `Tensor` of shape `(B, S, N, key_dim)`.
            value: Projected value `Tensor` of shape `(B, S, N, value_dim)`.
            attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
                attention to certain positions. It is generally not needed if
                the `query` and `value` (and/or `key`) are masked.
            training: Python boolean indicating whether the layer should behave
                in training mode (adding dropout) or in inference mode (doing
                nothing).

        Returns:
          attention_output: Multi-headed outputs of attention computation.
          attention_scores: Multi-headed attention weights.
        """
    query = tf.multiply(query, 1.0 / math.sqrt(float(self._key_dim)))
    attention_scores = tf.einsum(self._dot_product_equation, key, query)
    attention_scores = self._masked_softmax(attention_scores, attention_mask)
    attention_scores_dropout = self._dropout_layer(attention_scores, training=training)
    attention_output = tf.einsum(self._combine_equation, attention_scores_dropout, value)
    return (attention_output, attention_scores)
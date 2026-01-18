import tensorflow.compat.v2 as tf
from keras.src.layers.attention.base_dense_attention import BaseDenseAttention
from tensorflow.python.util.tf_export import keras_export
Calculates attention scores as a nonlinear sum of query and key.

        Args:
            query: Query tensor of shape `[batch_size, Tq, dim]`.
            key: Key tensor of shape `[batch_size, Tv, dim]`.
        Returns:
            Tensor of shape `[batch_size, Tq, Tv]`.
        
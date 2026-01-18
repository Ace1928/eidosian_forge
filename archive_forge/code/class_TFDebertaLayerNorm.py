from __future__ import annotations
import math
from typing import Dict, Optional, Sequence, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_deberta import DebertaConfig
class TFDebertaLayerNorm(keras.layers.Layer):
    """LayerNorm module in the TF style (epsilon inside the square root)."""

    def __init__(self, size, eps=1e-12, **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.eps = eps

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=[self.size], initializer=tf.ones_initializer(), name='weight')
        self.beta = self.add_weight(shape=[self.size], initializer=tf.zeros_initializer(), name='bias')
        return super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        std = tf.math.sqrt(variance + self.eps)
        return self.gamma * (x - mean) / std + self.beta
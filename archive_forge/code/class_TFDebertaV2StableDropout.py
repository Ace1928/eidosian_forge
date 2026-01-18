from __future__ import annotations
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_deberta_v2 import DebertaV2Config
class TFDebertaV2StableDropout(keras.layers.Layer):
    """
    Optimized dropout module for stabilizing the training

    Args:
        drop_prob (float): the dropout probabilities
    """

    def __init__(self, drop_prob, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    @tf.custom_gradient
    def xdropout(self, inputs):
        """
        Applies dropout to the inputs, as vanilla dropout, but also scales the remaining elements up by 1/drop_prob.
        """
        mask = tf.cast(1 - tf.compat.v1.distributions.Bernoulli(probs=1.0 - self.drop_prob).sample(sample_shape=shape_list(inputs)), tf.bool)
        scale = tf.convert_to_tensor(1.0 / (1 - self.drop_prob), dtype=tf.float32)
        if self.drop_prob > 0:
            inputs = tf.where(mask, 0.0, inputs) * scale

        def grad(upstream):
            if self.drop_prob > 0:
                return tf.where(mask, 0.0, upstream) * scale
            else:
                return upstream
        return (inputs, grad)

    def call(self, inputs: tf.Tensor, training: tf.Tensor=False):
        if training:
            return self.xdropout(inputs)
        return inputs
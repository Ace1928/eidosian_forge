from __future__ import annotations
import os
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, stable_softmax
from ...utils import logging
from .configuration_esm import EsmConfig
class TFEsmContactPredictionHead(keras.layers.Layer):
    """Performs symmetrization, apc, and computes a logistic regression on the output features"""

    def __init__(self, in_features: int, bias=True, eos_idx: int=2, name=None):
        super().__init__(name=name)
        self.eos_idx = eos_idx
        self.in_features = in_features
        self.regression = keras.layers.Dense(1, use_bias=bias, activation='sigmoid', name='regression')

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'regression', None) is not None:
            with tf.name_scope(self.regression.name):
                self.regression.build((None, self.in_features))

    def call(self, tokens, attentions):
        eos_mask = tf.cast(tokens != self.eos_idx, attentions.dtype)
        eos_mask = tf.expand_dims(eos_mask, 1) * tf.expand_dims(eos_mask, 2)
        attentions = attentions * eos_mask[:, None, None, :, :]
        attentions = attentions[..., :-1, :-1]
        attentions = attentions[..., 1:, 1:]
        batch_size, layers, heads, seqlen, _ = shape_list(attentions)
        attentions = tf.reshape(attentions, (batch_size, layers * heads, seqlen, seqlen))
        attentions = average_product_correct(symmetrize(attentions))
        attentions = tf.transpose(attentions, perm=(0, 2, 3, 1))
        return tf.squeeze(self.regression(attentions), 3)
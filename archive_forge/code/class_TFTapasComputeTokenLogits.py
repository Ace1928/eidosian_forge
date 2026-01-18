from __future__ import annotations
import enum
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_tapas import TapasConfig
class TFTapasComputeTokenLogits(keras.layers.Layer):

    def __init__(self, config: TapasConfig, **kwargs):
        super().__init__(**kwargs)
        self.temperature = config.temperature
        with tf.name_scope('output'):
            self.output_weights = self.add_weight(name='output_weights', shape=(config.hidden_size,), dtype=tf.float32, trainable=True, initializer=tf.zeros_initializer() if config.init_cell_selection_weights_to_zero else keras.initializers.TruncatedNormal(stddev=config.initializer_range))
            self.output_bias = self.add_weight(name='output_bias', shape=(), trainable=True, initializer=tf.zeros_initializer())

    def call(self, sequence_output: tf.Tensor) -> tf.Tensor:
        """
        Computes logits per token

        Args:
            sequence_output (`tf.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Also known as last_hidden_state. Sequence of hidden-states at the output of the last layer of the
                model.

        Returns:
            logits (`tf.Tensor` of shape `(batch_size, sequence_length)`): Logits per token.
        """
        logits = (tf.einsum('bsj,j->bs', sequence_output, self.output_weights) + self.output_bias) / self.temperature
        return logits
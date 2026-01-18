from __future__ import annotations
import warnings
from typing import Any, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFCausalLMOutput
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_hubert import HubertConfig
class TFHubertWeightNormConv1D(keras.layers.Conv1D):
    """Adapted from https://www.tensorflow.org/probability/api_docs/python/tfp/layers/weight_norm/WeightNorm"""

    def __init__(self, filters, kernel_size, groups, explicit_padding, **kwargs):
        super().__init__(filters=filters, kernel_size=kernel_size, groups=groups, padding='valid', use_bias=True, bias_initializer='he_normal', **kwargs)
        self.explicit_padding = explicit_padding
        self.filter_axis = 2
        self.kernel_norm_axes = tf.constant([0, 1])

    def _init_norm(self):
        """Set the norm of the weight vector."""
        kernel_norm = tf.sqrt(tf.reduce_sum(tf.square(self.weight_v), axis=self.kernel_norm_axes))
        self.weight_g.assign(kernel_norm[:, tf.newaxis, tf.newaxis])

    def _normalize_kernel(self):
        """Generate normalized weights."""
        kernel = tf.nn.l2_normalize(self.weight_v, axis=self.kernel_norm_axes) * tf.transpose(self.weight_g)
        self.kernel = tf.transpose(kernel)

    def build(self, input_shape):
        if not self.built:
            super().build(input_shape)
            self.kernel = tf.Variable(tf.transpose(self.kernel), name='weight_v', trainable=True)
            self.weight_v = self.kernel
            self.weight_g = self.add_weight(name='weight_g', shape=(int(self.weight_v.shape[self.filter_axis]), 1, 1), initializer='ones', dtype=self.weight_v.dtype, trainable=True)
            self._init_norm()
            self.bias = self.add_weight(name='bias', shape=(self.filters,), initializer='zeros', trainable=True)

    def call(self, inputs):
        self._normalize_kernel()
        padded_inputs = tf.pad(inputs, ((0, 0), (self.explicit_padding, self.explicit_padding), (0, 0)))
        output = super().call(padded_inputs)
        return output
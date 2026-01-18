from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_mobilebert import MobileBertConfig
class TFMobileBertAttention(keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.self = TFMobileBertSelfAttention(config, name='self')
        self.mobilebert_output = TFMobileBertSelfOutput(config, name='output')

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(self, query_tensor, key_tensor, value_tensor, layer_input, attention_mask, head_mask, output_attentions, training=False):
        self_outputs = self.self(query_tensor, key_tensor, value_tensor, attention_mask, head_mask, output_attentions, training=training)
        attention_output = self.mobilebert_output(self_outputs[0], layer_input, training=training)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'self', None) is not None:
            with tf.name_scope(self.self.name):
                self.self.build(None)
        if getattr(self, 'mobilebert_output', None) is not None:
            with tf.name_scope(self.mobilebert_output.name):
                self.mobilebert_output.build(None)
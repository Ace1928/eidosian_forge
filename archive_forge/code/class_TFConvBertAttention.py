from __future__ import annotations
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_convbert import ConvBertConfig
class TFConvBertAttention(keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.self_attention = TFConvBertSelfAttention(config, name='self')
        self.dense_output = TFConvBertSelfOutput(config, name='output')

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(self, input_tensor, attention_mask, head_mask, output_attentions, training=False):
        self_outputs = self.self_attention(input_tensor, attention_mask, head_mask, output_attentions, training=training)
        attention_output = self.dense_output(self_outputs[0], input_tensor, training=training)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'self_attention', None) is not None:
            with tf.name_scope(self.self_attention.name):
                self.self_attention.build(None)
        if getattr(self, 'dense_output', None) is not None:
            with tf.name_scope(self.dense_output.name):
                self.dense_output.build(None)
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
from .configuration_funnel import FunnelConfig
class TFFunnelLayer(keras.layers.Layer):

    def __init__(self, config, block_index, **kwargs):
        super().__init__(**kwargs)
        self.attention = TFFunnelRelMultiheadAttention(config, block_index, name='attention')
        self.ffn = TFFunnelPositionwiseFFN(config, name='ffn')

    def call(self, query, key, value, attention_inputs, output_attentions=False, training=False):
        attn = self.attention(query, key, value, attention_inputs, output_attentions=output_attentions, training=training)
        output = self.ffn(attn[0], training=training)
        return (output, attn[1]) if output_attentions else (output,)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'attention', None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        if getattr(self, 'ffn', None) is not None:
            with tf.name_scope(self.ffn.name):
                self.ffn.build(None)
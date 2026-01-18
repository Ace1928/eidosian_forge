from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutputWithPastAndCrossAttentions
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_led import LEDConfig
class TFLEDEncoderAttention(keras.layers.Layer):

    def __init__(self, config, layer_id, **kwargs):
        super().__init__(**kwargs)
        self.longformer_self_attn = TFLEDEncoderSelfAttention(config, layer_id=layer_id, name='longformer_self_attn')
        self.output_dense = keras.layers.Dense(config.d_model, use_bias=True, name='output')
        self.config = config

    def call(self, inputs, training=False):
        hidden_states, attention_mask, layer_head_mask, is_index_masked, is_index_global_attn, is_global_attn = inputs
        self_outputs = self.longformer_self_attn([hidden_states, attention_mask, layer_head_mask, is_index_masked, is_index_global_attn, is_global_attn], training=training)
        attention_output = self.output_dense(self_outputs[0], training=training)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'longformer_self_attn', None) is not None:
            with tf.name_scope(self.longformer_self_attn.name):
                self.longformer_self_attn.build(None)
        if getattr(self, 'output_dense', None) is not None:
            with tf.name_scope(self.output_dense.name):
                self.output_dense.build([None, None, self.config.d_model])
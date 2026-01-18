from __future__ import annotations
import math
import warnings
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_roberta import RobertaConfig
class TFRobertaAttention(keras.layers.Layer):

    def __init__(self, config: RobertaConfig, **kwargs):
        super().__init__(**kwargs)
        self.self_attention = TFRobertaSelfAttention(config, name='self')
        self.dense_output = TFRobertaSelfOutput(config, name='output')

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(self, input_tensor: tf.Tensor, attention_mask: tf.Tensor, head_mask: tf.Tensor, encoder_hidden_states: tf.Tensor, encoder_attention_mask: tf.Tensor, past_key_value: Tuple[tf.Tensor], output_attentions: bool, training: bool=False) -> Tuple[tf.Tensor]:
        self_outputs = self.self_attention(hidden_states=input_tensor, attention_mask=attention_mask, head_mask=head_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, past_key_value=past_key_value, output_attentions=output_attentions, training=training)
        attention_output = self.dense_output(hidden_states=self_outputs[0], input_tensor=input_tensor, training=training)
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
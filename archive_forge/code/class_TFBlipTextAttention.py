from __future__ import annotations
import math
from typing import Optional, Tuple
import tensorflow as tf
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, invert_attention_mask, stable_softmax
from ...utils import add_start_docstrings_to_model_forward, logging
from .configuration_blip import BlipTextConfig
class TFBlipTextAttention(keras.layers.Layer):

    def __init__(self, config, is_cross_attention=False, **kwargs):
        super().__init__(**kwargs)
        self.self = TFBlipTextSelfAttention(config, is_cross_attention, name='self')
        self.self_output = TFBlipTextSelfOutput(config, name='output')

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, encoder_hidden_states: tf.Tensor | None=None, encoder_attention_mask: tf.Tensor | None=None, past_key_value: Tuple[Tuple[tf.Tensor]] | None=None, output_attentions: Optional[bool]=False, training: Optional[bool]=None):
        self_outputs = self.self(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions, training=training)
        attention_output = self.self_output(self_outputs[0], hidden_states, training=training)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'self', None) is not None:
            with tf.name_scope(self.self.name):
                self.self.build(None)
        if getattr(self, 'self_output', None) is not None:
            with tf.name_scope(self.self_output.name):
                self.self_output.build(None)
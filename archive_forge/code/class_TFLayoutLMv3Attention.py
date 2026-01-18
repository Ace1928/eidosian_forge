from __future__ import annotations
import collections
import math
from typing import List, Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from .configuration_layoutlmv3 import LayoutLMv3Config
class TFLayoutLMv3Attention(keras.layers.Layer):

    def __init__(self, config: LayoutLMv3Config, **kwargs):
        super().__init__(**kwargs)
        self.self_attention = TFLayoutLMv3SelfAttention(config, name='self')
        self.self_output = TFLayoutLMv3SelfOutput(config, name='output')

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor | None, head_mask: tf.Tensor | None, output_attentions: bool, rel_pos: tf.Tensor | None=None, rel_2d_pos: tf.Tensor | None=None, training: bool=False) -> Union[Tuple[tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
        self_outputs = self.self_attention(hidden_states, attention_mask, head_mask, output_attentions, rel_pos, rel_2d_pos, training=training)
        attention_output = self.self_output(self_outputs[0], hidden_states, training=training)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'self_attention', None) is not None:
            with tf.name_scope(self.self_attention.name):
                self.self_attention.build(None)
        if getattr(self, 'self_output', None) is not None:
            with tf.name_scope(self.self_output.name):
                self.self_output.build(None)
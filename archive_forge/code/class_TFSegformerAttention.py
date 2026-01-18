from __future__ import annotations
import math
from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...file_utils import (
from ...modeling_tf_outputs import TFBaseModelOutput, TFSemanticSegmenterOutput, TFSequenceClassifierOutput
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import logging
from .configuration_segformer import SegformerConfig
class TFSegformerAttention(keras.layers.Layer):

    def __init__(self, config: SegformerConfig, hidden_size: int, num_attention_heads: int, sequence_reduction_ratio: int, **kwargs):
        super().__init__(**kwargs)
        self.self = TFSegformerEfficientSelfAttention(config=config, hidden_size=hidden_size, num_attention_heads=num_attention_heads, sequence_reduction_ratio=sequence_reduction_ratio, name='self')
        self.dense_output = TFSegformerSelfOutput(config, hidden_size=hidden_size, name='output')

    def call(self, hidden_states: tf.Tensor, height: int, width: int, output_attentions: bool=False) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        self_outputs = self.self(hidden_states, height, width, output_attentions)
        attention_output = self.dense_output(self_outputs[0])
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'self', None) is not None:
            with tf.name_scope(self.self.name):
                self.self.build(None)
        if getattr(self, 'dense_output', None) is not None:
            with tf.name_scope(self.dense_output.name):
                self.dense_output.build(None)
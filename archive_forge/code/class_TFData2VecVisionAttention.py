from __future__ import annotations
import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_data2vec_vision import Data2VecVisionConfig
class TFData2VecVisionAttention(keras.layers.Layer):

    def __init__(self, config: Data2VecVisionConfig, window_size: Optional[tuple]=None, **kwargs):
        super().__init__(**kwargs)
        self.attention = TFData2VecVisionSelfAttention(config, window_size=window_size, name='attention')
        self.dense_output = TFData2VecVisionSelfOutput(config, name='output')

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(self, input_tensor: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, relative_position_bias: Optional['TFData2VecVisionRelativePositionBias']=None, training: bool=False) -> Tuple[tf.Tensor]:
        self_outputs = self.attention(hidden_states=input_tensor, head_mask=head_mask, output_attentions=output_attentions, relative_position_bias=relative_position_bias, training=training)
        attention_output = self.dense_output(hidden_states=self_outputs[0], input_tensor=input_tensor, training=training)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'attention', None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        if getattr(self, 'dense_output', None) is not None:
            with tf.name_scope(self.dense_output.name):
                self.dense_output.build(None)
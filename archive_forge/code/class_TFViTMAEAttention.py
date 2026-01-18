from __future__ import annotations
import collections.abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...file_utils import (
from ...modeling_tf_outputs import TFBaseModelOutput
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import logging
from .configuration_vit_mae import ViTMAEConfig
class TFViTMAEAttention(keras.layers.Layer):

    def __init__(self, config: ViTMAEConfig, **kwargs):
        super().__init__(**kwargs)
        self.self_attention = TFViTMAESelfAttention(config, name='attention')
        self.dense_output = TFViTMAESelfOutput(config, name='output')

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(self, input_tensor: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, training: bool=False) -> Tuple[tf.Tensor]:
        self_outputs = self.self_attention(hidden_states=input_tensor, head_mask=head_mask, output_attentions=output_attentions, training=training)
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
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
class TFSegformerLayer(keras.layers.Layer):
    """This corresponds to the Block class in the original implementation."""

    def __init__(self, config, hidden_size: int, num_attention_heads: int, drop_path: float, sequence_reduction_ratio: int, mlp_ratio: int, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm_1 = keras.layers.LayerNormalization(epsilon=1e-05, name='layer_norm_1')
        self.attention = TFSegformerAttention(config, hidden_size=hidden_size, num_attention_heads=num_attention_heads, sequence_reduction_ratio=sequence_reduction_ratio, name='attention')
        self.drop_path = TFSegformerDropPath(drop_path) if drop_path > 0.0 else keras.layers.Activation('linear')
        self.layer_norm_2 = keras.layers.LayerNormalization(epsilon=1e-05, name='layer_norm_2')
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = TFSegformerMixFFN(config, in_features=hidden_size, hidden_features=mlp_hidden_size, name='mlp')
        self.hidden_size = hidden_size

    def call(self, hidden_states: tf.Tensor, height: int, width: int, output_attentions: bool=False, training: bool=False) -> Tuple:
        self_attention_outputs = self.attention(self.layer_norm_1(hidden_states), height, width, output_attentions=output_attentions, training=training)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        attention_output = self.drop_path(attention_output, training=training)
        hidden_states = attention_output + hidden_states
        mlp_output = self.mlp(self.layer_norm_2(hidden_states), height, width)
        mlp_output = self.drop_path(mlp_output, training=training)
        layer_output = mlp_output + hidden_states
        outputs = (layer_output,) + outputs
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'layer_norm_1', None) is not None:
            with tf.name_scope(self.layer_norm_1.name):
                self.layer_norm_1.build([None, None, self.hidden_size])
        if getattr(self, 'attention', None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        if getattr(self, 'layer_norm_2', None) is not None:
            with tf.name_scope(self.layer_norm_2.name):
                self.layer_norm_2.build([None, None, self.hidden_size])
        if getattr(self, 'mlp', None) is not None:
            with tf.name_scope(self.mlp.name):
                self.mlp.build(None)
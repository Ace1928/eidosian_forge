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
class TFData2VecVisionLayer(keras.layers.Layer):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: Data2VecVisionConfig, window_size: Optional[tuple]=None, drop_path_rate: float=0.0, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.attention = TFData2VecVisionAttention(config, window_size=window_size, name='attention')
        self.intermediate = TFData2VecVisionIntermediate(config, name='intermediate')
        self.data2vec_output = TFData2VecVisionOutput(config, name='output')
        self.layernorm_before = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layernorm_before')
        self.layernorm_after = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layernorm_after')
        self.drop_path = TFData2VecVisionDropPath(drop_path_rate, name='drop_path') if drop_path_rate > 0.0 else keras.layers.Activation('linear', name='drop_path')
        self.init_values = config.layer_scale_init_value

    def build(self, input_shape: tf.TensorShape=None):
        if self.init_values > 0:
            self.lambda_1 = self.add_weight(shape=self.config.hidden_size, initializer='ones', trainable=True, name='lambda_1')
            self.lambda_2 = self.add_weight(shape=self.config.hidden_size, initializer='ones', trainable=True, name='lambda_2')
            self.lambda_1.assign(self.init_values * tf.ones(self.config.hidden_size))
            self.lambda_2.assign(self.init_values * tf.ones(self.config.hidden_size))
        else:
            self.lambda_1, self.lambda_2 = (None, None)
        if self.built:
            return
        self.built = True
        if getattr(self, 'attention', None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        if getattr(self, 'intermediate', None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        if getattr(self, 'data2vec_output', None) is not None:
            with tf.name_scope(self.data2vec_output.name):
                self.data2vec_output.build(None)
        if getattr(self, 'layernorm_before', None) is not None:
            with tf.name_scope(self.layernorm_before.name):
                self.layernorm_before.build([None, None, self.config.hidden_size])
        if getattr(self, 'layernorm_after', None) is not None:
            with tf.name_scope(self.layernorm_after.name):
                self.layernorm_after.build([None, None, self.config.hidden_size])
        if getattr(self, 'drop_path', None) is not None:
            with tf.name_scope(self.drop_path.name):
                self.drop_path.build(None)

    def call(self, hidden_states: tf.Tensor, head_mask: tf.Tensor, output_attentions: bool, relative_position_bias: Optional['TFData2VecVisionRelativePositionBias']=None, training: bool=False) -> Tuple[tf.Tensor]:
        self_attention_outputs = self.attention(input_tensor=self.layernorm_before(inputs=hidden_states), head_mask=head_mask, output_attentions=output_attentions, relative_position_bias=relative_position_bias, training=training)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        if self.lambda_1 is not None:
            attention_output = self.lambda_1 * attention_output
        hidden_states = self.drop_path(attention_output) + hidden_states
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.data2vec_output(layer_output)
        if self.lambda_2 is not None:
            layer_output = self.lambda_2 * layer_output
        layer_output = self.drop_path(layer_output) + hidden_states
        outputs = (layer_output,) + outputs
        return outputs
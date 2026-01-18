from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFCausalLMOutput, TFSequenceClassifierOutput
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_openai import OpenAIGPTConfig
class TFBlock(keras.layers.Layer):

    def __init__(self, config, scale=False, **kwargs):
        super().__init__(**kwargs)
        nx = config.n_embd
        self.attn = TFAttention(nx, config, scale, name='attn')
        self.ln_1 = keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name='ln_1')
        self.mlp = TFMLP(4 * nx, config, name='mlp')
        self.ln_2 = keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name='ln_2')
        self.nx = nx

    def call(self, x, attention_mask, head_mask, output_attentions, training=False):
        output_attn = self.attn(x, attention_mask, head_mask, output_attentions, training=training)
        a = output_attn[0]
        n = self.ln_1(x + a)
        m = self.mlp(n, training=training)
        h = self.ln_2(n + m)
        outputs = [h] + output_attn[1:]
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'attn', None) is not None:
            with tf.name_scope(self.attn.name):
                self.attn.build(None)
        if getattr(self, 'ln_1', None) is not None:
            with tf.name_scope(self.ln_1.name):
                self.ln_1.build([None, None, self.nx])
        if getattr(self, 'mlp', None) is not None:
            with tf.name_scope(self.mlp.name):
                self.mlp.build(None)
        if getattr(self, 'ln_2', None) is not None:
            with tf.name_scope(self.ln_2.name):
                self.ln_2.build([None, None, self.nx])
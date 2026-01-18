from __future__ import annotations
import math
from typing import Optional, Tuple
import tensorflow as tf
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, invert_attention_mask, stable_softmax
from ...utils import add_start_docstrings_to_model_forward, logging
from .configuration_blip import BlipTextConfig
class TFBlipTextLMPredictionHead(keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.transform = TFBlipTextPredictionHeadTransform(config, name='transform')
        self.decoder = keras.layers.Dense(config.vocab_size, kernel_initializer=get_initializer(config.initializer_range), name='decoder', use_bias=False)
        self.config = config

    def build(self, input_shape=None):
        self.bias = self.add_weight(name='bias', shape=(self.config.vocab_size,), initializer='zeros', trainable=True)
        if self.built:
            return
        self.built = True
        if getattr(self, 'transform', None) is not None:
            with tf.name_scope(self.transform.name):
                self.transform.build(None)
        if getattr(self, 'decoder', None) is not None:
            with tf.name_scope(self.decoder.name):
                self.decoder.build([None, None, self.config.hidden_size])

    def call(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states
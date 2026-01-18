from __future__ import annotations
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...modeling_tf_outputs import TFBaseModelOutputWithPast, TFCausalLMOutputWithPast, TFSequenceClassifierOutput
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_ctrl import CTRLConfig
class TFPointWiseFeedForwardLayer(keras.layers.Layer):

    def __init__(self, d_model_size, dff, **kwargs):
        super().__init__(**kwargs)
        self.dense_0 = keras.layers.Dense(dff, activation='relu', name='0')
        self.dense_2 = keras.layers.Dense(d_model_size, name='2')
        self.d_model_size = d_model_size
        self.dff = dff

    def call(self, inputs, trainable=False):
        dense_0_output = self.dense_0(inputs)
        dense_2_output = self.dense_2(dense_0_output)
        return dense_2_output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'dense_0', None) is not None:
            with tf.name_scope(self.dense_0.name):
                self.dense_0.build([None, None, self.d_model_size])
        if getattr(self, 'dense_2', None) is not None:
            with tf.name_scope(self.dense_2.name):
                self.dense_2.build([None, None, self.dff])
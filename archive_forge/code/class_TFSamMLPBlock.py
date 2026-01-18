from __future__ import annotations
import collections
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...modeling_tf_outputs import TFBaseModelOutput
from ...modeling_tf_utils import TFModelInputType, TFPreTrainedModel, keras, shape_list, unpack_inputs
from ...tf_utils import flatten, functional_layernorm
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_sam import SamConfig, SamMaskDecoderConfig, SamPromptEncoderConfig, SamVisionConfig
class TFSamMLPBlock(keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.lin1 = keras.layers.Dense(config.mlp_dim, name='lin1')
        self.lin2 = keras.layers.Dense(config.hidden_size, name='lin2')
        self.act = ACT2FN[config.hidden_act]
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.lin1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.lin2(hidden_states)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'lin1', None) is not None:
            with tf.name_scope(self.lin1.name):
                self.lin1.build([None, None, self.config.hidden_size])
        if getattr(self, 'lin2', None) is not None:
            with tf.name_scope(self.lin2.name):
                self.lin2.build([None, None, self.config.mlp_dim])
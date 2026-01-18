from __future__ import annotations
import warnings
from typing import Any, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFCausalLMOutput
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_hubert import HubertConfig
class TFHubertPositionalConvEmbedding(keras.layers.Layer):

    def __init__(self, config: HubertConfig, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.conv = TFHubertWeightNormConv1D(filters=config.hidden_size, kernel_size=config.num_conv_pos_embeddings, groups=config.num_conv_pos_embedding_groups, explicit_padding=config.num_conv_pos_embeddings // 2, name='conv')
        self.padding = TFHubertSamePadLayer(config.num_conv_pos_embeddings)
        self.activation = get_tf_activation(config.feat_extract_activation)
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'conv', None) is not None:
            with tf.name_scope(self.conv.name):
                self.conv.build([None, None, self.config.hidden_size])
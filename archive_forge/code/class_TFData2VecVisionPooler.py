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
class TFData2VecVisionPooler(keras.layers.Layer):

    def __init__(self, config: Data2VecVisionConfig, **kwargs):
        super().__init__(**kwargs)
        self.layernorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layernorm') if config.use_mean_pooling else None
        self.config = config

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        if self.layernorm is not None:
            patch_tokens = hidden_states[:, 1:, :]
            pooled_output = self.layernorm(tf.reduce_mean(patch_tokens, axis=1))
        else:
            pooled_output = hidden_states[:, 0]
        return pooled_output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'layernorm', None) is not None:
            if hasattr(self.layernorm, 'name'):
                with tf.name_scope(self.layernorm.name):
                    self.layernorm.build((None, self.config.hidden_size))
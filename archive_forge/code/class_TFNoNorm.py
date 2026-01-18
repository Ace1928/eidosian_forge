from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_mobilebert import MobileBertConfig
class TFNoNorm(keras.layers.Layer):

    def __init__(self, feat_size, epsilon=None, **kwargs):
        super().__init__(**kwargs)
        self.feat_size = feat_size

    def build(self, input_shape):
        self.bias = self.add_weight('bias', shape=[self.feat_size], initializer='zeros')
        self.weight = self.add_weight('weight', shape=[self.feat_size], initializer='ones')
        super().build(input_shape)

    def call(self, inputs: tf.Tensor):
        return inputs * self.weight + self.bias
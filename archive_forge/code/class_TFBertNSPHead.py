from __future__ import annotations
import math
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_bert import BertConfig
class TFBertNSPHead(keras.layers.Layer):

    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)
        self.seq_relationship = keras.layers.Dense(units=2, kernel_initializer=get_initializer(config.initializer_range), name='seq_relationship')
        self.config = config

    def call(self, pooled_output: tf.Tensor) -> tf.Tensor:
        seq_relationship_score = self.seq_relationship(inputs=pooled_output)
        return seq_relationship_score

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'seq_relationship', None) is not None:
            with tf.name_scope(self.seq_relationship.name):
                self.seq_relationship.build([None, None, self.config.hidden_size])
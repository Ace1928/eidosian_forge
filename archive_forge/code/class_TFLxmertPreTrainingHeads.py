from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, stable_softmax
from ...utils import (
from .configuration_lxmert import LxmertConfig
class TFLxmertPreTrainingHeads(keras.layers.Layer):

    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)
        self.predictions = TFLxmertLMPredictionHead(config, input_embeddings, name='predictions')
        self.seq_relationship = keras.layers.Dense(2, kernel_initializer=get_initializer(config.initializer_range), name='seq_relationship')
        self.config = config

    def call(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return (prediction_scores, seq_relationship_score)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'predictions', None) is not None:
            with tf.name_scope(self.predictions.name):
                self.predictions.build(None)
        if getattr(self, 'seq_relationship', None) is not None:
            with tf.name_scope(self.seq_relationship.name):
                self.seq_relationship.build([None, None, self.config.hidden_size])
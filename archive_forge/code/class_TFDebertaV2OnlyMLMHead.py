from __future__ import annotations
from typing import Dict, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_deberta_v2 import DebertaV2Config
class TFDebertaV2OnlyMLMHead(keras.layers.Layer):

    def __init__(self, config: DebertaV2Config, input_embeddings: keras.layers.Layer, **kwargs):
        super().__init__(**kwargs)
        self.predictions = TFDebertaV2LMPredictionHead(config, input_embeddings, name='predictions')

    def call(self, sequence_output: tf.Tensor) -> tf.Tensor:
        prediction_scores = self.predictions(hidden_states=sequence_output)
        return prediction_scores

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'predictions', None) is not None:
            with tf.name_scope(self.predictions.name):
                self.predictions.build(None)
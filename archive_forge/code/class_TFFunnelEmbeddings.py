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
from .configuration_funnel import FunnelConfig
class TFFunnelEmbeddings(keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.hidden_size = config.hidden_size
        self.initializer_std = 1.0 if config.initializer_std is None else config.initializer_std
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layer_norm')
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout)

    def build(self, input_shape=None):
        with tf.name_scope('word_embeddings'):
            self.weight = self.add_weight(name='weight', shape=[self.config.vocab_size, self.hidden_size], initializer=get_initializer(initializer_range=self.initializer_std))
        if self.built:
            return
        self.built = True
        if getattr(self, 'LayerNorm', None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.d_model])

    def call(self, input_ids=None, inputs_embeds=None, training=False):
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        assert not (input_ids is None and inputs_embeds is None)
        assert not (input_ids is not None and inputs_embeds is not None)
        if input_ids is not None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = tf.gather(self.weight, input_ids)
        final_embeddings = self.LayerNorm(inputs=inputs_embeds)
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)
        return final_embeddings
from __future__ import annotations
import warnings
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_distilbert import DistilBertConfig
@keras_serializable
class TFDistilBertMainLayer(keras.layers.Layer):
    config_class = DistilBertConfig

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.return_dict = config.use_return_dict
        self.embeddings = TFEmbeddings(config, name='embeddings')
        self.transformer = TFTransformer(config, name='transformer')

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings.weight = value
        self.embeddings.vocab_size = value.shape[0]

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError

    @unpack_inputs
    def call(self, input_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None, training=False):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if attention_mask is None:
            attention_mask = tf.ones(input_shape)
        attention_mask = tf.cast(attention_mask, dtype=tf.float32)
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.num_hidden_layers
        embedding_output = self.embeddings(input_ids, inputs_embeds=inputs_embeds)
        tfmr_output = self.transformer(embedding_output, attention_mask, head_mask, output_attentions, output_hidden_states, return_dict, training=training)
        return tfmr_output

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'embeddings', None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        if getattr(self, 'transformer', None) is not None:
            with tf.name_scope(self.transformer.name):
                self.transformer.build(None)
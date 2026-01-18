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
@keras_serializable
class TFFunnelMainLayer(keras.layers.Layer):
    """Base model with decoder"""
    config_class = FunnelConfig

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.block_sizes = config.block_sizes
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.return_dict = config.use_return_dict
        self.embeddings = TFFunnelEmbeddings(config, name='embeddings')
        self.encoder = TFFunnelEncoder(config, name='encoder')
        self.decoder = TFFunnelDecoder(config, name='decoder')

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError

    @unpack_inputs
    def call(self, input_ids=None, attention_mask=None, token_type_ids=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None, training=False):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if attention_mask is None:
            attention_mask = tf.fill(input_shape, 1)
        if token_type_ids is None:
            token_type_ids = tf.fill(input_shape, 0)
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids, training=training)
        encoder_outputs = self.encoder(inputs_embeds, attention_mask=attention_mask, token_type_ids=token_type_ids, output_attentions=output_attentions, output_hidden_states=True, return_dict=return_dict, training=training)
        decoder_outputs = self.decoder(final_hidden=encoder_outputs[0], first_block_hidden=encoder_outputs[1][self.block_sizes[0]], attention_mask=attention_mask, token_type_ids=token_type_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        if not return_dict:
            idx = 0
            outputs = (decoder_outputs[0],)
            if output_hidden_states:
                idx += 1
                outputs = outputs + (encoder_outputs[1] + decoder_outputs[idx],)
            if output_attentions:
                idx += 1
                outputs = outputs + (encoder_outputs[2] + decoder_outputs[idx],)
            return outputs
        return TFBaseModelOutput(last_hidden_state=decoder_outputs[0], hidden_states=encoder_outputs.hidden_states + decoder_outputs.hidden_states if output_hidden_states else None, attentions=encoder_outputs.attentions + decoder_outputs.attentions if output_attentions else None)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'embeddings', None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        if getattr(self, 'encoder', None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        if getattr(self, 'decoder', None) is not None:
            with tf.name_scope(self.decoder.name):
                self.decoder.build(None)
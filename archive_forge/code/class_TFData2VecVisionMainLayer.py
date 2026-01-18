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
@keras_serializable
class TFData2VecVisionMainLayer(keras.layers.Layer):
    config_class = Data2VecVisionConfig

    def __init__(self, config: Data2VecVisionConfig, add_pooling_layer: bool=True, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.add_pooling_layer = add_pooling_layer
        self.embeddings = TFData2VecVisionEmbeddings(config, name='embeddings')
        self.encoder = TFData2VecVisionEncoder(config, window_size=self.embeddings.patch_embeddings.patch_shape, name='encoder')
        self.layernorm = tf.identity if config.use_mean_pooling else keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layernorm')
        self.pooler = TFData2VecVisionPooler(config, name='pooler') if add_pooling_layer else None

    def get_input_embeddings(self) -> keras.layers.Layer:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    @unpack_inputs
    def call(self, pixel_values: tf.Tensor | None=None, bool_masked_pos: tf.Tensor | None=None, head_mask: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[tuple, TFData2VecVisionModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.config.num_hidden_layers
        embedding_output = self.embeddings(pixel_values, bool_masked_pos, training=training)
        encoder_outputs = self.encoder(embedding_output, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]
        return TFData2VecVisionModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

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
        if getattr(self, 'layernorm', None) is not None:
            if hasattr(self.layernorm, 'name'):
                with tf.name_scope(self.layernorm.name):
                    self.layernorm.build((None, self.config.hidden_size))
        if getattr(self, 'pooler', None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build(None)
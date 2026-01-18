from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
class TFCLIPVisionTransformer(keras.layers.Layer):

    def __init__(self, config: CLIPVisionConfig, **kwargs):
        super().__init__(**kwargs)
        self.embeddings = TFCLIPVisionEmbeddings(config, name='embeddings')
        self.pre_layernorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='pre_layrnorm')
        self.encoder = TFCLIPEncoder(config, name='encoder')
        self.post_layernorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='post_layernorm')
        self.embed_dim = config.hidden_size

    def call(self, pixel_values: TFModelInputType, output_attentions: bool, output_hidden_states: bool, return_dict: bool, training: bool=False) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        embedding_output = self.embeddings(pixel_values=pixel_values)
        embedding_output = self.pre_layernorm(inputs=embedding_output)
        encoder_outputs = self.encoder(hidden_states=embedding_output, attention_mask=None, causal_attention_mask=None, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = encoder_outputs[0]
        pooled_output = sequence_output[:, 0, :]
        pooled_output = self.post_layernorm(inputs=pooled_output)
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        return TFBaseModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'embeddings', None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        if getattr(self, 'pre_layernorm', None) is not None:
            with tf.name_scope(self.pre_layernorm.name):
                self.pre_layernorm.build([None, None, self.embed_dim])
        if getattr(self, 'encoder', None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        if getattr(self, 'post_layernorm', None) is not None:
            with tf.name_scope(self.post_layernorm.name):
                self.post_layernorm.build([None, self.embed_dim])
from __future__ import annotations
import warnings
from typing import Any, Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import TFBaseModelOutput, TFCausalLMOutput
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_hubert import HubertConfig
class TFHubertEncoderStableLayerNorm(keras.layers.Layer):

    def __init__(self, config: HubertConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.pos_conv_embed = TFHubertPositionalConvEmbedding(config, name='pos_conv_embed')
        self.layer_norm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='layer_norm')
        self.dropout = keras.layers.Dropout(config.hidden_dropout)
        self.layer = [TFHubertEncoderLayerStableLayerNorm(config, name=f'layers.{i}') for i in range(config.num_hidden_layers)]

    def call(self, hidden_states: tf.Tensor, attention_mask: tf.Tensor | None=None, output_attentions: Optional[bool]=False, output_hidden_states: Optional[bool]=False, return_dict: Optional[bool]=True, training: Optional[bool]=False) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        if attention_mask is not None:
            hidden_states = hidden_states * tf.expand_dims(attention_mask, -1)
            attention_mask = _expand_mask(attention_mask)
        else:
            attention_mask = None
        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states, training=training)
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            dropout_probability = np.random.uniform(0, 1)
            if training and dropout_probability < self.config.layerdrop:
                continue
            layer_outputs = layer_module(hidden_states=hidden_states, attention_mask=attention_mask, output_attentions=output_attentions, training=training)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None))
        return TFBaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'pos_conv_embed', None) is not None:
            with tf.name_scope(self.pos_conv_embed.name):
                self.pos_conv_embed.build(None)
        if getattr(self, 'layer_norm', None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.hidden_size])
        if getattr(self, 'layer', None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Union
import tensorflow as tf
from ...modeling_tf_outputs import TFBaseModelOutputWithPooling
from ...modeling_tf_utils import TFModelInputType, TFPreTrainedModel, get_initializer, keras, shape_list, unpack_inputs
from ...utils import (
from ..bert.modeling_tf_bert import TFBertMainLayer
from .configuration_dpr import DPRConfig
class TFDPREncoderLayer(keras.layers.Layer):
    base_model_prefix = 'bert_model'

    def __init__(self, config: DPRConfig, **kwargs):
        super().__init__(**kwargs)
        self.bert_model = TFBertMainLayer(config, add_pooling_layer=False, name='bert_model')
        self.config = config
        if self.config.hidden_size <= 0:
            raise ValueError("Encoder hidden_size can't be zero")
        self.projection_dim = config.projection_dim
        if self.projection_dim > 0:
            self.encode_proj = keras.layers.Dense(config.projection_dim, kernel_initializer=get_initializer(config.initializer_range), name='encode_proj')

    @unpack_inputs
    def call(self, input_ids: tf.Tensor=None, attention_mask: tf.Tensor | None=None, token_type_ids: tf.Tensor | None=None, inputs_embeds: tf.Tensor | None=None, output_attentions: bool=None, output_hidden_states: bool=None, return_dict: bool=None, training: bool=False) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor, ...]]:
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0, :]
        if self.projection_dim > 0:
            pooled_output = self.encode_proj(pooled_output)
        if not return_dict:
            return (sequence_output, pooled_output) + outputs[1:]
        return TFBaseModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    @property
    def embeddings_size(self) -> int:
        if self.projection_dim > 0:
            return self.projection_dim
        return self.bert_model.config.hidden_size

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'bert_model', None) is not None:
            with tf.name_scope(self.bert_model.name):
                self.bert_model.build(None)
        if getattr(self, 'encode_proj', None) is not None:
            with tf.name_scope(self.encode_proj.name):
                self.encode_proj.build(None)
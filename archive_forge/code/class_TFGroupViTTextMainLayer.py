from __future__ import annotations
import collections.abc
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
from .configuration_groupvit import GroupViTConfig, GroupViTTextConfig, GroupViTVisionConfig
@keras_serializable
class TFGroupViTTextMainLayer(keras.layers.Layer):
    config_class = GroupViTTextConfig

    def __init__(self, config: GroupViTTextConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.text_model = TFGroupViTTextTransformer(config, name='text_model')

    def get_input_embeddings(self) -> keras.layers.Layer:
        return self.text_model.embeddings

    def set_input_embeddings(self, value: tf.Variable):
        self.text_model.embeddings.weight = value
        self.text_model.embeddings.vocab_size = shape_list(value)[0]

    @unpack_inputs
    def call(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFBaseModelOutputWithPooling, Tuple[tf.Tensor]]:
        if input_ids is None:
            raise ValueError('You have to specify input_ids')
        input_shape = shape_list(input_ids)
        if attention_mask is None:
            attention_mask = tf.fill(dims=input_shape, value=1)
        text_model_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return text_model_outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'text_model', None) is not None:
            with tf.name_scope(self.text_model.name):
                self.text_model.build(None)
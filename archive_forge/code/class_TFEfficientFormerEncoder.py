import itertools
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import shape_list, stable_softmax
from ...utils import (
from .configuration_efficientformer import EfficientFormerConfig
class TFEfficientFormerEncoder(keras.layers.Layer):

    def __init__(self, config: EfficientFormerConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        num_intermediate_stages = len(config.depths) - 1
        downsamples = [config.downsamples[i] or config.hidden_sizes[i] != config.hidden_sizes[i + 1] for i in range(num_intermediate_stages)]
        intermediate_stages = []
        layer_count = -1
        for i in range(num_intermediate_stages):
            layer_count += 1
            intermediate_stages.append(TFEfficientFormerIntermediateStage(config, i, name=f'intermediate_stages.{layer_count}'))
            if downsamples[i]:
                layer_count += 1
                intermediate_stages.append(TFEfficientFormerPatchEmbeddings(config, config.hidden_sizes[i], config.hidden_sizes[i + 1], name=f'intermediate_stages.{layer_count}'))
        self.intermediate_stages = intermediate_stages
        self.last_stage = TFEfficientFormerLastStage(config, name='last_stage')

    def call(self, hidden_states: tf.Tensor, output_hidden_states: bool, output_attentions: bool, return_dict: bool, training: bool=False) -> TFBaseModelOutput:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        for layer_module in self.intermediate_stages:
            hidden_states = layer_module(hidden_states, training=training)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
        layer_output = self.last_stage(hidden_states, output_attentions=output_attentions, training=training)
        if output_attentions:
            all_self_attentions = all_self_attentions + layer_output[1:]
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (layer_output[0],)
        if not return_dict:
            return tuple((v for v in [layer_output[0], all_hidden_states, all_self_attentions] if v is not None))
        return TFBaseModelOutput(last_hidden_state=layer_output[0], hidden_states=all_hidden_states, attentions=all_self_attentions)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'last_stage', None) is not None:
            with tf.name_scope(self.last_stage.name):
                self.last_stage.build(None)
        for layer in self.intermediate_stages:
            with tf.name_scope(layer.name):
                layer.build(None)
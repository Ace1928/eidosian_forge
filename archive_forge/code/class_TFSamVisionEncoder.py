from __future__ import annotations
import collections
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import ACT2FN
from ...modeling_tf_outputs import TFBaseModelOutput
from ...modeling_tf_utils import TFModelInputType, TFPreTrainedModel, keras, shape_list, unpack_inputs
from ...tf_utils import flatten, functional_layernorm
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_sam import SamConfig, SamMaskDecoderConfig, SamPromptEncoderConfig, SamVisionConfig
class TFSamVisionEncoder(keras.layers.Layer):

    def __init__(self, config: SamVisionConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.image_size = config.image_size
        self.patch_embed = TFSamPatchEmbeddings(config, name='patch_embed')
        self.pos_embed = None
        self.layers = []
        for i in range(config.num_hidden_layers):
            layer = TFSamVisionLayer(config, window_size=config.window_size if i not in config.global_attn_indexes else 0, name=f'layers_._{i}')
            self.layers.append(layer)
        self.neck = TFSamVisionNeck(config, name='neck')

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if self.config.use_abs_pos:
            self.pos_embed = self.add_weight(shape=[1, self.config.image_size // self.config.patch_size, self.config.image_size // self.config.patch_size, self.config.hidden_size], initializer='zeros', trainable=True, name='pos_embed')
        if getattr(self, 'patch_embed', None) is not None:
            with tf.name_scope(self.patch_embed.name):
                self.patch_embed.build(None)
        if getattr(self, 'neck', None) is not None:
            with tf.name_scope(self.neck.name):
                self.neck.build(None)
        for layer in self.layers:
            with tf.name_scope(layer.name):
                layer.build(None)

    def get_input_embeddings(self):
        return self.patch_embed

    def call(self, pixel_values: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=False) -> Union[Tuple, TFSamVisionEncoderOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        hidden_states = self.patch_embed(pixel_values)
        if self.pos_embed is not None:
            hidden_states = hidden_states + self.pos_embed
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states, output_attentions=output_attentions, training=training)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        hidden_states = self.neck(hidden_states)
        if not return_dict:
            outputs = (hidden_states,)
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if output_attentions:
                outputs = outputs + (all_self_attentions,)
            return outputs
        return TFSamVisionEncoderOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions)
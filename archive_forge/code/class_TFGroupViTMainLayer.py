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
class TFGroupViTMainLayer(keras.layers.Layer):
    config_class = GroupViTConfig

    def __init__(self, config: GroupViTConfig, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(config.text_config, GroupViTTextConfig):
            raise ValueError(f'config.text_config is expected to be of type GroupViTTextConfig but is of type {type(config.text_config)}.')
        if not isinstance(config.vision_config, GroupViTVisionConfig):
            raise ValueError(f'config.vision_config is expected to be of type GroupViTVisionConfig but is of type {type(config.vision_config)}.')
        self.config = config
        text_config = config.text_config
        vision_config = config.vision_config
        self.projection_dim = config.projection_dim
        self.projection_intermediate_dim = config.projection_intermediate_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size
        self.text_model = TFGroupViTTextTransformer(text_config, name='text_model')
        self.vision_model = TFGroupViTVisionTransformer(vision_config, name='vision_model')
        self.visual_projection = [keras.layers.Dense(self.projection_intermediate_dim, name='visual_projection.0'), keras.layers.BatchNormalization(name='visual_projection.1', momentum=0.9, epsilon=1e-05), keras.layers.ReLU(name='visual_projection.2'), keras.layers.Dense(self.projection_dim, name='visual_projection.3')]
        self.text_projection = [keras.layers.Dense(self.projection_intermediate_dim, name='text_projection.0'), keras.layers.BatchNormalization(name='text_projection.1', momentum=0.9, epsilon=1e-05), keras.layers.ReLU(name='text_projection.2'), keras.layers.Dense(self.projection_dim, name='text_projection.3')]

    def build(self, input_shape=None):
        self.logit_scale = self.add_weight(shape=(1,), initializer=keras.initializers.Constant(self.config.logit_scale_init_value), trainable=True, name='logit_scale')
        if self.built:
            return
        self.built = True
        if getattr(self, 'text_model', None) is not None:
            with tf.name_scope(self.text_model.name):
                self.text_model.build(None)
        if getattr(self, 'vision_model', None) is not None:
            with tf.name_scope(self.vision_model.name):
                self.vision_model.build(None)
        if getattr(self, 'visual_projection', None) is not None:
            with tf.name_scope(self.visual_projection[0].name):
                self.visual_projection[0].build([None, None, None, self.vision_embed_dim])
            with tf.name_scope(self.visual_projection[1].name):
                self.visual_projection[1].build((None, self.projection_intermediate_dim))
            with tf.name_scope(self.visual_projection[3].name):
                self.visual_projection[3].build([None, None, None, self.projection_intermediate_dim])
        if getattr(self, 'text_projection', None) is not None:
            with tf.name_scope(self.text_projection[0].name):
                self.text_projection[0].build([None, None, None, self.text_embed_dim])
            with tf.name_scope(self.text_projection[1].name):
                self.text_projection[1].build((None, self.projection_intermediate_dim))
            with tf.name_scope(self.text_projection[3].name):
                self.text_projection[3].build([None, None, None, self.projection_intermediate_dim])

    @unpack_inputs
    def get_text_features(self, input_ids: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> tf.Tensor:
        if input_ids is None:
            raise ValueError('You have to specify either input_ids')
        input_shape = shape_list(input_ids)
        if attention_mask is None:
            attention_mask = tf.fill(dims=input_shape, value=1)
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        pooled_output = text_outputs[1]
        for layer in self.text_projection:
            pooled_output = layer(pooled_output)
        text_features = pooled_output
        return text_features

    @unpack_inputs
    def get_image_features(self, pixel_values: TFModelInputType | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> tf.Tensor:
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        pooled_output = vision_outputs[1]
        for layer in self.visual_projection:
            pooled_output = layer(pooled_output)
        image_features = pooled_output
        return image_features

    @unpack_inputs
    def call(self, input_ids: TFModelInputType | None=None, pixel_values: TFModelInputType | None=None, attention_mask: np.ndarray | tf.Tensor | None=None, position_ids: np.ndarray | tf.Tensor | None=None, return_loss: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, output_segmentation: Optional[bool]=None, return_dict: Optional[bool]=None, training: bool=False) -> Union[TFGroupViTModelOutput, Tuple[tf.Tensor]]:
        if input_ids is None:
            raise ValueError('You have to specify either input_ids')
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        input_shape = shape_list(input_ids)
        if attention_mask is None:
            attention_mask = tf.fill(dims=input_shape, value=1)
        if output_segmentation:
            output_attentions = True
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        image_embeds = vision_outputs[1]
        for layer in self.visual_projection:
            image_embeds = layer(image_embeds)
        text_embeds = text_outputs[1]
        for layer in self.text_projection:
            text_embeds = layer(text_embeds)
        image_embeds = image_embeds / tf.norm(image_embeds, axis=-1, keepdims=True)
        text_embeds = text_embeds / tf.norm(text_embeds, axis=-1, keepdims=True)
        logit_scale = tf.math.exp(self.logit_scale)
        logits_per_text = tf.matmul(text_embeds, image_embeds, transpose_b=True) * logit_scale
        logits_per_image = tf.transpose(logits_per_text)
        seg_logits = None
        if output_segmentation:
            image_group_embeds = vision_outputs[0]
            image_group_embeds = tf.reshape(image_group_embeds, shape=(-1, shape_list(image_group_embeds)[-1]))
            for layer in self.visual_projection:
                image_group_embeds = layer(image_group_embeds)
            if output_hidden_states:
                attentions = vision_outputs[3]
            else:
                attentions = vision_outputs[2]
            grouping = get_grouping_from_attentions(attentions, pixel_values.shape[2:])
            image_group_embeds = image_group_embeds / tf.norm(tensor=image_group_embeds, ord='euclidean', axis=-1, keepdims=True)
            logits_per_image_group = tf.matmul(image_group_embeds, text_embeds, transpose_b=True) * logit_scale
            logits_per_image_group = tf.reshape(logits_per_image_group, shape=(image_embeds.shape[0], -1, text_embeds.shape[0]))
            logits_per_image_group = tf.transpose(logits_per_image_group, perm=(0, 2, 1))
            flatten_grouping = tf.reshape(grouping, shape=(shape_list(grouping)[0], shape_list(grouping)[1], -1))
            seg_logits = tf.matmul(logits_per_image_group, flatten_grouping) * logit_scale
            seg_logits = tf.reshape(seg_logits, shape=(seg_logits.shape[0], seg_logits.shape[1], grouping.shape[2], grouping.shape[3]))
        loss = None
        if return_loss:
            loss = groupvit_loss(logits_per_text)[None, ...]
        if not return_dict:
            if seg_logits is not None:
                output = (logits_per_image, logits_per_text, seg_logits, text_embeds, image_embeds, text_outputs, vision_outputs)
            else:
                output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return (loss,) + output if loss is not None else output
        return TFGroupViTModelOutput(loss=loss, logits_per_image=logits_per_image, logits_per_text=logits_per_text, segmentation_logits=seg_logits, text_embeds=text_embeds, image_embeds=image_embeds, text_model_output=text_outputs, vision_model_output=vision_outputs)
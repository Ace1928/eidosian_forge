from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import tensorflow as tf
from ...modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, stable_softmax
from ...utils import (
from .configuration_blip import BlipConfig, BlipTextConfig, BlipVisionConfig
from .modeling_tf_blip_text import BLIP_TEXT_INPUTS_DOCSTRING, TFBlipTextLMHeadModel, TFBlipTextModel
class TFBlipMainLayer(keras.layers.Layer):
    config_class = BlipConfig

    def __init__(self, config: BlipConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(config.text_config, BlipTextConfig):
            raise ValueError(f'config.text_config is expected to be of type BlipTextConfig but is of type {type(config.text_config)}.')
        if not isinstance(config.vision_config, BlipVisionConfig):
            raise ValueError(f'config.vision_config is expected to be of type BlipVisionConfig but is of type {type(config.vision_config)}.')
        text_config = config.text_config
        vision_config = config.vision_config
        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size
        self.text_model = TFBlipTextModel(text_config, name='text_model')
        self.vision_model = TFBlipVisionModel(vision_config, name='vision_model')
        self.visual_projection = keras.layers.Dense(self.projection_dim, use_bias=False, kernel_initializer=get_initializer(config.initializer_range), name='visual_projection')
        self.text_projection = keras.layers.Dense(self.projection_dim, use_bias=False, kernel_initializer=get_initializer(config.initializer_range), name='text_projection')
        self.config = config

    def build(self, input_shape=None):
        self.logit_scale = self.add_weight(name='logit_scale', shape=[], initializer=keras.initializers.Constant(self.config.logit_scale_init_value), trainable=True)
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
            with tf.name_scope(self.visual_projection.name):
                self.visual_projection.build([None, None, self.vision_embed_dim])
        if getattr(self, 'text_projection', None) is not None:
            with tf.name_scope(self.text_projection.name):
                self.text_projection.build([None, None, self.text_embed_dim])

    @unpack_inputs
    def call(self, input_ids: tf.Tensor | None=None, pixel_values: tf.Tensor | None=None, attention_mask: tf.Tensor | None=None, position_ids: tf.Tensor | None=None, return_loss: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=None) -> Union[Tuple, TFBlipOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)
        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)
        image_embeds = image_embeds / tf.norm(image_embeds, ord=2, axis=-1, keepdims=True)
        text_embeds = text_embeds / tf.norm(text_embeds, ord=2, axis=-1, keepdims=True)
        logit_scale = tf.exp(self.logit_scale)
        logits_per_text = tf.matmul(text_embeds, image_embeds, transpose_b=True) * logit_scale
        logits_per_image = tf.transpose(logits_per_text)
        loss = None
        if return_loss:
            loss = blip_loss(logits_per_text)
            loss = tf.reshape(loss, (1,))
        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return (loss,) + output if loss is not None else output
        return TFBlipOutput(loss=loss, logits_per_image=logits_per_image, logits_per_text=logits_per_text, text_embeds=text_embeds, image_embeds=image_embeds, text_model_output=text_outputs, vision_model_output=vision_outputs)
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
@add_start_docstrings('\n    BLIP Model with a vision and text projector, and a classification head on top. The model is used in the context of\n    image-text retrieval. Given an image and a text, the model returns the probability of the text being relevant to\n    the image.\n    ', BLIP_START_DOCSTRING)
class TFBlipForImageTextRetrieval(TFBlipPreTrainedModel):
    config_class = BlipConfig

    def __init__(self, config: BlipConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.vision_model = TFBlipVisionModel(config.vision_config, name='vision_model')
        self.text_encoder = TFBlipTextModel(config.text_config, name='text_encoder', add_pooling_layer=False)
        self.vision_proj = keras.layers.Dense(config.image_text_hidden_size, kernel_initializer=get_initializer(config.initializer_range), name='vision_proj')
        self.text_proj = keras.layers.Dense(config.image_text_hidden_size, kernel_initializer=get_initializer(config.initializer_range), name='text_proj')
        self.itm_head = keras.layers.Dense(2, kernel_initializer=get_initializer(config.initializer_range), name='itm_head')
        self.decoder_pad_token_id = config.text_config.pad_token_id if not hasattr(config, 'decoder_pad_token_id') else config.decoder_pad_token_id
        self.decoder_start_token_id = config.text_config.bos_token_id if not hasattr(config, 'decoder_start_token_id') else config.decoder_start_token_id
        self.config = config

    def get_input_embeddings(self) -> keras.layers.Layer:
        return self.vision_model.embeddings.patch_embedding

    @unpack_inputs
    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBlipImageTextMatchingModelOutput, config_class=BlipVisionConfig)
    def call(self, input_ids: tf.Tensor, pixel_values: tf.Tensor | None=None, use_itm_head: Optional[bool]=True, attention_mask: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=None) -> Union[Tuple, TFBlipImageTextMatchingModelOutput]:
        """
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFBlipForImageTextRetrieval

        >>> model = TFBlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-base-coco")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "an image of a cat"

        >>> inputs = processor(images=image, text=text, return_tensors="tf")
        >>> outputs = model(**inputs)
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        image_embeds = vision_outputs[0]
        image_atts = tf.ones(shape_list(image_embeds)[:-1], dtype=tf.int64)
        itm_question_embeds = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, encoder_hidden_states=image_embeds, encoder_attention_mask=image_atts, return_dict=return_dict, training=training)
        itm_question_embeds = itm_question_embeds[0] if not return_dict else itm_question_embeds.last_hidden_state
        itm_output = self.itm_head(itm_question_embeds[:, 0, :])
        no_itm_question_embeds = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict, training=training)
        no_itm_question_embeds = no_itm_question_embeds[0] if not return_dict else no_itm_question_embeds.last_hidden_state
        image_feat, _ = tf.linalg.normalize(self.vision_proj(image_embeds[:, 0, :]), ord=2, axis=-1)
        text_feat, _ = tf.linalg.normalize(self.text_proj(no_itm_question_embeds[:, 0, :]), ord=2, axis=-1)
        no_itm_output = tf.matmul(image_feat, text_feat, transpose_b=True)
        if use_itm_head:
            output = itm_output
            question_embeds = itm_question_embeds
        else:
            output = no_itm_output
            question_embeds = no_itm_question_embeds
        if not return_dict:
            outputs = (output, vision_outputs[0]) + vision_outputs[2:] + (question_embeds,)
            return tuple((output for output in outputs if output is not None))
        return TFBlipImageTextMatchingModelOutput(itm_score=output, last_hidden_state=vision_outputs.last_hidden_state, hidden_states=vision_outputs.hidden_states, attentions=vision_outputs.attentions, question_embeds=question_embeds)

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'vision_model', None) is not None:
            with tf.name_scope(self.vision_model.name):
                self.vision_model.build(None)
        if getattr(self, 'text_encoder', None) is not None:
            with tf.name_scope(self.text_encoder.name):
                self.text_encoder.build(None)
        if getattr(self, 'vision_proj', None) is not None:
            with tf.name_scope(self.vision_proj.name):
                self.vision_proj.build([None, None, self.config.vision_config.hidden_size])
        if getattr(self, 'text_proj', None) is not None:
            with tf.name_scope(self.text_proj.name):
                self.text_proj.build([None, None, self.config.text_config.hidden_size])
        if getattr(self, 'itm_head', None) is not None:
            with tf.name_scope(self.itm_head.name):
                self.itm_head.build([None, None, self.config.text_config.hidden_size])
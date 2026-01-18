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
@add_start_docstrings('\n    BLIP Model for visual question answering. The model consists of a vision encoder, a text encoder as well as a text\n    decoder. The vision encoder will encode the input image, the text encoder will encode the input question together\n    with the encoding of the image, and the text decoder will output the answer to the question.\n    ', BLIP_START_DOCSTRING)
class TFBlipForQuestionAnswering(TFBlipPreTrainedModel):
    config_class = BlipConfig
    _keys_to_ignore_on_load_missing = ['text_decoder.cls.predictions.decoder.bias']

    def __init__(self, config: BlipConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.vision_model = TFBlipVisionModel(config.vision_config, name='vision_model')
        self.text_encoder = TFBlipTextModel(config.text_config, name='text_encoder', add_pooling_layer=False)
        self.text_decoder = TFBlipTextLMHeadModel(config.text_config, name='text_decoder')
        self.decoder_pad_token_id = config.text_config.pad_token_id
        self.decoder_start_token_id = config.text_config.bos_token_id

    def get_input_embeddings(self) -> keras.layers.Layer:
        return self.vision_model.embeddings.patch_embedding

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.decoder_start_token_id
        pad_token_id = self.decoder_pad_token_id
        if decoder_start_token_id is None or pad_token_id is None:
            raise ValueError('decoder_start_token_id and pad_token_id must be defined!')
        start_tokens = tf.fill((shape_list(input_ids)[0], 1), decoder_start_token_id)
        start_tokens = tf.cast(start_tokens, input_ids.dtype)
        shifted_input_ids = tf.concat([start_tokens, input_ids[:, :-1]], -1)
        shifted_input_ids = tf.where(shifted_input_ids == -100, tf.cast(tf.fill(shape_list(shifted_input_ids), pad_token_id), shifted_input_ids.dtype), shifted_input_ids)
        tf.debugging.assert_greater_equal(shifted_input_ids, tf.constant(0, dtype=shifted_input_ids.dtype))
        return shifted_input_ids

    @unpack_inputs
    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBlipTextVisionModelOutput, config_class=BlipVisionConfig)
    def call(self, input_ids: tf.Tensor, pixel_values: tf.Tensor | None=None, decoder_input_ids: tf.Tensor | None=None, decoder_attention_mask: tf.Tensor | None=None, attention_mask: tf.Tensor | None=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, labels: tf.Tensor | None=None, return_dict: Optional[bool]=None, training: Optional[bool]=None) -> Union[Tuple, TFBlipTextVisionModelOutput]:
        """
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFBlipForQuestionAnswering

        >>> model = TFBlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> # training
        >>> text = "How many cats are in the picture?"
        >>> label = "2"
        >>> inputs = processor(images=image, text=text, return_tensors="tf")
        >>> labels = processor(text=label, return_tensors="tf").input_ids

        >>> inputs["labels"] = labels
        >>> outputs = model(**inputs)
        >>> loss = outputs.loss

        >>> # inference
        >>> text = "How many cats are in the picture?"
        >>> inputs = processor(images=image, text=text, return_tensors="tf")
        >>> outputs = model.generate(**inputs)
        >>> print(processor.decode(outputs[0], skip_special_tokens=True))
        2
        ```"""
        if labels is None and decoder_input_ids is None:
            raise ValueError('Either `decoder_input_ids` or `labels` should be passed when calling `TFBlipForQuestionAnswering`. if you are training the model make sure that `labels` is passed, if you are using the model for inference make sure that `decoder_input_ids` is passed or call `generate`')
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        image_embeds = vision_outputs[0]
        image_attention_mask = tf.ones(shape_list(image_embeds)[:-1], dtype=tf.int64)
        question_embeds = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, encoder_hidden_states=image_embeds, encoder_attention_mask=image_attention_mask, return_dict=return_dict, training=training)
        question_embeds = question_embeds[0] if not return_dict else question_embeds.last_hidden_state
        if labels is not None and decoder_input_ids is None:
            decoder_input_ids = labels
        answer_output = self.text_decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, encoder_hidden_states=question_embeds, encoder_attention_mask=attention_mask, labels=labels, return_dict=return_dict, training=training)
        if labels is not None:
            decoder_loss = tf.reduce_mean(answer_output.loss) if return_dict else tf.reduce_mean(answer_output[0])
        else:
            decoder_loss = None
        if not return_dict:
            outputs = (decoder_loss, image_embeds, vision_outputs[0]) + vision_outputs[2:]
            return tuple((output for output in outputs if output is not None))
        return TFBlipTextVisionModelOutput(loss=decoder_loss, image_embeds=image_embeds, last_hidden_state=vision_outputs.last_hidden_state, hidden_states=vision_outputs.hidden_states, attentions=vision_outputs.attentions)

    def generate(self, input_ids: tf.Tensor, pixel_values: tf.Tensor, attention_mask: tf.Tensor | None=None, **generate_kwargs) -> tf.Tensor:
        """
        Overrides *generate* function to be able to use the model as a conditional generator

        Parameters:
            input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            pixel_values (`tf.Tensor` of shape `(batch_size, num_channels, image_height, image_width)`:
                Input image to be processed
            attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`. `1` for
                tokens that are NOT MASKED, `0` for MASKED tokens.
            generate_kwargs (dict, *optional*):
                Additional arguments passed to the `generate` function of the decoder


        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFBlipForQuestionAnswering

        >>> model = TFBlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "How many cats are in the picture?"

        >>> inputs = processor(images=image, text=text, return_tensors="tf")

        >>> outputs = model.generate(**inputs)
        >>> print(processor.decode(outputs[0], skip_special_tokens=True))
        2
        ```
        """
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        image_embeds = vision_outputs[0]
        image_attention_mask = tf.ones(shape_list(image_embeds)[:-1], dtype=tf.int32)
        if isinstance(input_ids, list):
            input_ids = tf.Tensor(input_ids)
        question_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, encoder_hidden_states=image_embeds, encoder_attention_mask=image_attention_mask, return_dict=False)
        question_embeds = question_outputs[0]
        question_attention_mask = tf.ones(shape_list(question_embeds)[:-1], dtype=tf.int32)
        bos_ids = tf.fill((tf.shape(question_embeds)[0], 1), value=tf.cast(self.decoder_start_token_id, input_ids.dtype))
        outputs = self.text_decoder.generate(input_ids=bos_ids, eos_token_id=self.config.text_config.sep_token_id, pad_token_id=self.config.text_config.pad_token_id, encoder_hidden_states=question_embeds, encoder_attention_mask=question_attention_mask, **generate_kwargs)
        return outputs

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
        if getattr(self, 'text_decoder', None) is not None:
            with tf.name_scope(self.text_decoder.name):
                self.text_decoder.build(None)
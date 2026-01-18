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
class TFBlipModel(TFBlipPreTrainedModel):
    config_class = BlipConfig
    _keys_to_ignore_on_load_missing = ['text_decoder.cls.predictions.decoder.bias']
    main_input_name = 'input_ids'

    def __init__(self, config: BlipConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.blip = TFBlipMainLayer(config, name='blip')

    def serving_output(self, output: TFBlipOutput) -> TFBlipOutput:
        return TFBlipOutput(logits_per_image=output.logits_per_image, logits_per_text=output.logits_per_text, text_embeds=output.text_embeds, image_embeds=output.image_embeds)

    @unpack_inputs
    @add_start_docstrings_to_model_forward(BLIP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBlipOutput, config_class=BlipConfig)
    def call(self, input_ids: tf.Tensor | None=None, pixel_values: tf.Tensor | None=None, attention_mask: tf.Tensor | None=None, position_ids: tf.Tensor | None=None, return_loss: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, training: Optional[bool]=None) -> Union[Tuple, TFBlipOutput]:
        """
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFBlipModel

        >>> model = TFBlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="tf", padding=True
        ... )

        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = tf.nn.softmax(logits_per_image, axis=1)  # we can take the softmax to get the label probabilities
        ```"""
        outputs = self.blip(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, position_ids=position_ids, return_loss=return_loss, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        return outputs

    @add_start_docstrings_to_model_forward(BLIP_TEXT_INPUTS_DOCSTRING)
    def get_text_features(self, input_ids: tf.Tensor | None=None, attention_mask: tf.Tensor | None=None, position_ids: tf.Tensor | None=None, return_dict: Optional[bool]=None) -> tf.Tensor:
        """
        Returns:
            text_features (`tf.Tensor` of shape `(batch_size, output_dim`): The text embeddings obtained by applying
            the projection layer to the pooled output of [`TFBlipTextModel`].

        Examples:

        ```python
        >>> from transformers import AutoProcessor, TFBlipModel

        >>> model = TFBlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

        >>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="tf")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        text_outputs = self.blip.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, return_dict=return_dict)
        pooled_output = text_outputs[1]
        text_features = self.blip.text_projection(pooled_output)
        return text_features

    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    def get_image_features(self, pixel_values: tf.Tensor | None=None, return_dict: Optional[bool]=None) -> tf.Tensor:
        """
        Returns:
            image_features (`tf.Tensor` of shape `(batch_size, output_dim`): The image embeddings obtained by applying
            the projection layer to the pooled output of [`TFBlipVisionModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, TFBlipModel

        >>> model = TFBlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="tf")

        >>> image_features = model.get_image_features(**inputs)
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_outputs = self.blip.vision_model(pixel_values=pixel_values, return_dict=return_dict)
        pooled_output = vision_outputs[1]
        image_features = self.blip.visual_projection(pooled_output)
        return image_features

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'blip', None) is not None:
            with tf.name_scope(self.blip.name):
                self.blip.build(None)
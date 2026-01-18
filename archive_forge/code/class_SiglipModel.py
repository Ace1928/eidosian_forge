import math
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.init import _calculate_fan_in_and_fan_out
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_siglip import SiglipConfig, SiglipTextConfig, SiglipVisionConfig
@add_start_docstrings(SIGLIP_START_DOCSTRING)
class SiglipModel(SiglipPreTrainedModel):
    config_class = SiglipConfig

    def __init__(self, config: SiglipConfig):
        super().__init__(config)
        if not isinstance(config.text_config, SiglipTextConfig):
            raise ValueError(f'config.text_config is expected to be of type SiglipTextConfig but is of type {type(config.text_config)}.')
        if not isinstance(config.vision_config, SiglipVisionConfig):
            raise ValueError(f'config.vision_config is expected to be of type SiglipVisionConfig but is of type {type(config.vision_config)}.')
        text_config = config.text_config
        vision_config = config.vision_config
        self.text_model = SiglipTextTransformer(text_config)
        self.vision_model = SiglipVisionTransformer(vision_config)
        self.logit_scale = nn.Parameter(torch.randn(1))
        self.logit_bias = nn.Parameter(torch.randn(1))
        self.post_init()

    @add_start_docstrings_to_model_forward(SIGLIP_TEXT_INPUTS_DOCSTRING)
    def get_text_features(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> torch.FloatTensor:
        """
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`SiglipTextModel`].

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, AutoModel
        >>> import torch

        >>> model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")

        >>> # important: make sure to set padding="max_length" as that's how the model was trained
        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding="max_length", return_tensors="pt")
        >>> with torch.no_grad():
        ...     text_features = model.get_text_features(**inputs)
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        pooled_output = text_outputs[1]
        return pooled_output

    @add_start_docstrings_to_model_forward(SIGLIP_VISION_INPUTS_DOCSTRING)
    def get_image_features(self, pixel_values: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> torch.FloatTensor:
        """
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`SiglipVisionModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, AutoModel
        >>> import torch

        >>> model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        >>> processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     image_features = model.get_image_features(**inputs)
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        pooled_output = vision_outputs[1]
        return pooled_output

    @add_start_docstrings_to_model_forward(SIGLIP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SiglipOutput, config_class=SiglipConfig)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, pixel_values: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, return_loss: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, SiglipOutput]:
        """
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, AutoModel
        >>> import torch

        >>> model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        >>> processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> texts = ["a photo of 2 cats", "a photo of 2 dogs"]
        >>> # important: we pass `padding=max_length` since the model was trained with this
        >>> inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> logits_per_image = outputs.logits_per_image
        >>> probs = torch.sigmoid(logits_per_image) # these are the probabilities
        >>> print(f"{probs[0][0]:.1%} that image 0 is '{texts[0]}'")
        31.9% that image 0 is 'a photo of 2 cats'
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        image_embeds = vision_outputs[1]
        text_embeds = text_outputs[1]
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * self.logit_scale.exp() + self.logit_bias
        logits_per_image = logits_per_text.t()
        loss = None
        if return_loss:
            raise NotImplementedError('SigLIP loss to be implemented')
        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return (loss,) + output if loss is not None else output
        return SiglipOutput(loss=loss, logits_per_image=logits_per_image, logits_per_text=logits_per_text, text_embeds=text_embeds, image_embeds=image_embeds, text_model_output=text_outputs, vision_model_output=vision_outputs)
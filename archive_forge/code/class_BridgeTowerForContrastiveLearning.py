import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN, QuickGELUActivation
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel, apply_chunking_to_forward
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_bridgetower import BridgeTowerConfig, BridgeTowerTextConfig, BridgeTowerVisionConfig
@add_start_docstrings('\n    BridgeTower Model with a image-text contrastive head on top computing image-text contrastive loss.\n    ', BRIDGETOWER_START_DOCSTRING)
class BridgeTowerForContrastiveLearning(BridgeTowerPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.bridgetower = BridgeTowerModel(config)
        self.itc_text_head = BridgeTowerContrastiveHead(config.hidden_size, config.contrastive_hidden_size)
        self.itc_image_head = BridgeTowerContrastiveHead(config.hidden_size, config.contrastive_hidden_size)
        self.itc_cross_modal_head = BridgeTowerContrastiveHead(config.hidden_size * 2, config.contrastive_hidden_size)
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))
        self.post_init()

    @add_start_docstrings_to_model_forward(BRIDGETOWER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BridgeTowerContrastiveOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, pixel_values: Optional[torch.FloatTensor]=None, pixel_mask: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, image_embeds: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=True, return_dict: Optional[bool]=None, return_loss: Optional[bool]=None) -> Union[BridgeTowerContrastiveOutput, Tuple[torch.FloatTensor]]:
        """
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.
        Returns:

        Examples:

        ```python
        >>> from transformers import BridgeTowerProcessor, BridgeTowerForContrastiveLearning
        >>> import requests
        >>> from PIL import Image
        >>> import torch

        >>> image_urls = [
        ...     "https://farm4.staticflickr.com/3395/3428278415_81c3e27f15_z.jpg",
        ...     "http://images.cocodataset.org/val2017/000000039769.jpg",
        ... ]
        >>> texts = ["two dogs in a car", "two cats sleeping on a couch"]
        >>> images = [Image.open(requests.get(url, stream=True).raw) for url in image_urls]

        >>> processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")
        >>> model = BridgeTowerForContrastiveLearning.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")

        >>> inputs = processor(images, texts, padding=True, return_tensors="pt")
        >>> loss = model(**inputs, return_loss=True).loss

        >>> inputs = processor(images, texts[::-1], padding=True, return_tensors="pt")
        >>> loss_swapped = model(**inputs, return_loss=True).loss

        >>> print("Loss", round(loss.item(), 4))
        Loss 0.0019

        >>> print("Loss with swapped images", round(loss_swapped.item(), 4))
        Loss with swapped images 2.126
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bridgetower(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, pixel_values=pixel_values, pixel_mask=pixel_mask, head_mask=head_mask, inputs_embeds=inputs_embeds, image_embeds=image_embeds, output_attentions=output_attentions, output_hidden_states=True, return_dict=return_dict)
        pooler_output = outputs.pooler_output if return_dict else outputs[2]
        hidden_states_txt, hidden_states_img, hidden_states_cross_modal = outputs.hidden_states if return_dict else outputs[3]
        text_embeds = hidden_states_txt[-1]
        image_embeds = hidden_states_img[-1]
        image_embeds_with_ln = self.bridgetower.vision_model.visual.forward_post(image_embeds)
        image_token_type_embeddings = self.bridgetower.token_type_embeddings(torch.full((1,), 1, dtype=torch.long, device=self.bridgetower.token_type_embeddings.weight.device)).expand_as(image_embeds_with_ln)
        image_embeds = self.bridgetower.cross_modal_image_transform(image_embeds_with_ln) + image_token_type_embeddings
        text_embeds = nn.functional.normalize(self.itc_text_head(text_embeds[:, 0, :]), dim=-1, p=2)
        image_embeds = nn.functional.normalize(self.itc_image_head(image_embeds[:, 0, :]), dim=-1, p=2).to(device=text_embeds.device)
        cross_embeds = nn.functional.normalize(self.itc_cross_modal_head(pooler_output), dim=-1, p=2).to(device=text_embeds.device)
        logits = torch.stack([text_embeds, image_embeds, cross_embeds], dim=-2)
        logit_scale = self.logit_scale.exp().to(device=text_embeds.device)
        logits_text_to_image = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_text_to_cross = torch.matmul(text_embeds, cross_embeds.t()) * logit_scale
        logits_image_to_cross = torch.matmul(image_embeds, cross_embeds.t()) * logit_scale
        itc_loss = None
        if return_loss:
            labels = torch.arange(len(logits), device=logits.device)
            text_to_image_loss = nn.functional.cross_entropy(logits_text_to_image, labels)
            text_to_cross_loss = nn.functional.cross_entropy(logits_text_to_cross, labels)
            image_to_cross_loss = nn.functional.cross_entropy(logits_image_to_cross, labels)
            itc_loss = (text_to_image_loss + text_to_cross_loss + image_to_cross_loss) / 3.0
        if not return_dict:
            output = (logits, text_embeds, image_embeds, cross_embeds) + outputs[3:]
            return (itc_loss,) + output if itc_loss is not None else output
        return BridgeTowerContrastiveOutput(loss=itc_loss, logits=logits, text_embeds=text_embeds, image_embeds=image_embeds, cross_embeds=cross_embeds, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
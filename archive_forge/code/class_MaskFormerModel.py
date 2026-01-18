import math
from dataclasses import dataclass
from numbers import Number
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from ..detr import DetrConfig
from .configuration_maskformer import MaskFormerConfig
from .configuration_maskformer_swin import MaskFormerSwinConfig
@add_start_docstrings('The bare MaskFormer Model outputting raw hidden-states without any specific head on top.', MASKFORMER_START_DOCSTRING)
class MaskFormerModel(MaskFormerPreTrainedModel):

    def __init__(self, config: MaskFormerConfig):
        super().__init__(config)
        self.pixel_level_module = MaskFormerPixelLevelModule(config)
        self.transformer_module = MaskFormerTransformerModule(in_features=self.pixel_level_module.encoder.channels[-1], config=config)
        self.post_init()

    @add_start_docstrings_to_model_forward(MASKFORMER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MaskFormerModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: Tensor, pixel_mask: Optional[Tensor]=None, output_hidden_states: Optional[bool]=None, output_attentions: Optional[bool]=None, return_dict: Optional[bool]=None) -> MaskFormerModelOutput:
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, MaskFormerModel
        >>> from PIL import Image
        >>> import requests

        >>> # load MaskFormer fine-tuned on ADE20k semantic segmentation
        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-ade")
        >>> model = MaskFormerModel.from_pretrained("facebook/maskformer-swin-base-ade")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = image_processor(image, return_tensors="pt")

        >>> # forward pass
        >>> outputs = model(**inputs)

        >>> # the decoder of MaskFormer outputs hidden states of shape (batch_size, num_queries, hidden_size)
        >>> transformer_decoder_last_hidden_state = outputs.transformer_decoder_last_hidden_state
        >>> list(transformer_decoder_last_hidden_state.shape)
        [1, 100, 256]
        ```"""
        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size, _, height, width = pixel_values.shape
        if pixel_mask is None:
            pixel_mask = torch.ones((batch_size, height, width), device=pixel_values.device)
        pixel_level_module_output = self.pixel_level_module(pixel_values, output_hidden_states, return_dict=return_dict)
        image_features = pixel_level_module_output[0]
        pixel_embeddings = pixel_level_module_output[1]
        transformer_module_output = self.transformer_module(image_features, output_hidden_states, output_attentions)
        queries = transformer_module_output.last_hidden_state
        encoder_hidden_states = None
        pixel_decoder_hidden_states = None
        transformer_decoder_hidden_states = None
        hidden_states = None
        if output_hidden_states:
            encoder_hidden_states = pixel_level_module_output[2]
            pixel_decoder_hidden_states = pixel_level_module_output[3]
            transformer_decoder_hidden_states = transformer_module_output[1]
            hidden_states = encoder_hidden_states + pixel_decoder_hidden_states + transformer_decoder_hidden_states
        output = MaskFormerModelOutput(encoder_last_hidden_state=image_features, pixel_decoder_last_hidden_state=pixel_embeddings, transformer_decoder_last_hidden_state=queries, encoder_hidden_states=encoder_hidden_states, pixel_decoder_hidden_states=pixel_decoder_hidden_states, transformer_decoder_hidden_states=transformer_decoder_hidden_states, hidden_states=hidden_states, attentions=transformer_module_output.attentions)
        if not return_dict:
            output = tuple((v for v in output.values()))
        return output
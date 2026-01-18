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
class MaskFormerPixelLevelModule(nn.Module):

    def __init__(self, config: MaskFormerConfig):
        """
        Pixel Level Module proposed in [Per-Pixel Classification is Not All You Need for Semantic
        Segmentation](https://arxiv.org/abs/2107.06278). It runs the input image through a backbone and a pixel
        decoder, generating an image feature map and pixel embeddings.

        Args:
            config ([`MaskFormerConfig`]):
                The configuration used to instantiate this model.
        """
        super().__init__()
        if getattr(config, 'backbone_config') is not None and config.backbone_config.model_type == 'swin':
            backbone_config = config.backbone_config
            backbone_config = MaskFormerSwinConfig.from_dict(backbone_config.to_dict())
            backbone_config.out_features = ['stage1', 'stage2', 'stage3', 'stage4']
            config.backbone_config = backbone_config
        self.encoder = load_backbone(config)
        feature_channels = self.encoder.channels
        self.decoder = MaskFormerPixelDecoder(in_features=feature_channels[-1], feature_size=config.fpn_feature_size, mask_feature_size=config.mask_feature_size, lateral_widths=feature_channels[:-1])

    def forward(self, pixel_values: Tensor, output_hidden_states: bool=False, return_dict: bool=True) -> MaskFormerPixelLevelModuleOutput:
        features = self.encoder(pixel_values).feature_maps
        decoder_output = self.decoder(features, output_hidden_states, return_dict=return_dict)
        if not return_dict:
            last_hidden_state = decoder_output[0]
            outputs = (features[-1], last_hidden_state)
            if output_hidden_states:
                hidden_states = decoder_output[1]
                outputs = outputs + (tuple(features),) + (hidden_states,)
            return outputs
        return MaskFormerPixelLevelModuleOutput(encoder_last_hidden_state=features[-1], decoder_last_hidden_state=decoder_output.last_hidden_state, encoder_hidden_states=tuple(features) if output_hidden_states else (), decoder_hidden_states=decoder_output.hidden_states if output_hidden_states else ())
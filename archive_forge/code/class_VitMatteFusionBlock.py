from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import nn
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from .configuration_vitmatte import VitMatteConfig
class VitMatteFusionBlock(nn.Module):
    """
    Simple fusion block to fuse features from ConvStream and Plain Vision Transformer.
    """

    def __init__(self, config, in_channels, out_channels):
        super().__init__()
        self.conv = VitMatteBasicConv3x3(config, in_channels, out_channels, stride=1, padding=1)

    def forward(self, features, detailed_feature_map):
        upscaled_features = nn.functional.interpolate(features, scale_factor=2, mode='bilinear', align_corners=False)
        out = torch.cat([detailed_feature_map, upscaled_features], dim=1)
        out = self.conv(out)
        return out
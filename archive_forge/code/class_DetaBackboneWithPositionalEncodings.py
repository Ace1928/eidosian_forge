import copy
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from ...activations import ACT2FN
from ...file_utils import (
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import meshgrid
from ...utils import is_accelerate_available, is_ninja_available, is_torchvision_available, logging, requires_backends
from ...utils.backbone_utils import load_backbone
from .configuration_deta import DetaConfig
class DetaBackboneWithPositionalEncodings(nn.Module):
    """
    Backbone model with positional embeddings.

    nn.BatchNorm2d layers are replaced by DetaFrozenBatchNorm2d as defined above.
    """

    def __init__(self, config):
        super().__init__()
        backbone = load_backbone(config)
        with torch.no_grad():
            replace_batch_norm(backbone)
        self.model = backbone
        self.intermediate_channel_sizes = self.model.channels
        if config.backbone_config.model_type == 'resnet':
            for name, parameter in self.model.named_parameters():
                if 'stages.1' not in name and 'stages.2' not in name and ('stages.3' not in name):
                    parameter.requires_grad_(False)
        self.position_embedding = build_position_encoding(config)

    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):
        """
        Outputs feature maps of latter stages C_3 through C_5 in ResNet if `config.num_feature_levels > 1`, otherwise
        outputs feature maps of C_5.
        """
        features = self.model(pixel_values).feature_maps
        out = []
        pos = []
        for feature_map in features:
            mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
            position_embeddings = self.position_embedding(feature_map, mask).to(feature_map.dtype)
            out.append((feature_map, mask))
            pos.append(position_embeddings)
        return (out, pos)
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import prune_linear_layer
from ...utils import logging
from ...utils.backbone_utils import load_backbone
from .configuration_tvp import TvpConfig
class TvpVisionModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.backbone = load_backbone(config)
        self.grid_encoder_conv = nn.Conv2d(config.backbone_config.hidden_sizes[-1], config.hidden_size, kernel_size=3, stride=1, padding=1, groups=1, bias=False)

    def forward(self, pixel_values):
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.view(batch_size * num_frames, num_channels, height, width)
        grid_feat_outputs = self.backbone(pixel_values)['feature_maps'][0]
        grid = self.grid_encoder_conv(grid_feat_outputs)
        grid = nn.functional.max_pool2d(grid, kernel_size=2, stride=2)
        grid = nn.functional.relu(grid, inplace=True)
        new_channel, new_height, new_width = grid.shape[-3:]
        grid = grid.view(batch_size, num_frames, new_channel, new_height, new_width)
        grid = grid.permute(0, 1, 3, 4, 2)
        return grid
import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import Tensor, nn
from ...activations import ACT2FN
from ...file_utils import ModelOutput
from ...modeling_outputs import BackboneOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils.backbone_utils import BackboneMixin
from .configuration_maskformer_swin import MaskFormerSwinConfig
class MaskFormerSwinStage(nn.Module):

    def __init__(self, config, dim, input_resolution, depth, num_heads, drop_path, downsample):
        super().__init__()
        self.config = config
        self.dim = dim
        self.blocks = nn.ModuleList([MaskFormerSwinLayer(config=config, dim=dim, input_resolution=input_resolution, num_heads=num_heads, shift_size=0 if i % 2 == 0 else config.window_size // 2) for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=nn.LayerNorm)
        else:
            self.downsample = None
        self.pointing = False

    def forward(self, hidden_states, input_dimensions, head_mask=None, output_attentions=False, output_hidden_states=False):
        all_hidden_states = () if output_hidden_states else None
        height, width = input_dimensions
        for i, block_module in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            block_hidden_states = block_module(hidden_states, input_dimensions, layer_head_mask, output_attentions)
            hidden_states = block_hidden_states[0]
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
        if self.downsample is not None:
            height_downsampled, width_downsampled = ((height + 1) // 2, (width + 1) // 2)
            output_dimensions = (height, width, height_downsampled, width_downsampled)
            hidden_states = self.downsample(hidden_states, input_dimensions)
        else:
            output_dimensions = (height, width, height, width)
        return (hidden_states, output_dimensions, all_hidden_states)
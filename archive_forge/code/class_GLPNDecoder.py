import math
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, DepthEstimatorOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_glpn import GLPNConfig
class GLPNDecoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        reserved_hidden_sizes = config.hidden_sizes[::-1]
        out_channels = config.decoder_hidden_size
        self.stages = nn.ModuleList([GLPNDecoderStage(hidden_size, out_channels) for hidden_size in reserved_hidden_sizes])
        self.stages[0].fusion = None
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, hidden_states: List[torch.Tensor]) -> List[torch.Tensor]:
        stage_hidden_states = []
        stage_hidden_state = None
        for hidden_state, stage in zip(hidden_states[::-1], self.stages):
            stage_hidden_state = stage(hidden_state, stage_hidden_state)
            stage_hidden_states.append(stage_hidden_state)
        stage_hidden_states[-1] = self.final_upsample(stage_hidden_state)
        return stage_hidden_states
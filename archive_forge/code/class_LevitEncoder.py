import itertools
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_levit import LevitConfig
class LevitEncoder(nn.Module):
    """
    LeViT Encoder consisting of multiple `LevitStage` stages.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        resolution = self.config.image_size // self.config.patch_size
        self.stages = []
        self.config.down_ops.append([''])
        for stage_idx in range(len(config.depths)):
            stage = LevitStage(config, stage_idx, config.hidden_sizes[stage_idx], config.key_dim[stage_idx], config.depths[stage_idx], config.num_attention_heads[stage_idx], config.attention_ratio[stage_idx], config.mlp_ratio[stage_idx], config.down_ops[stage_idx], resolution)
            resolution = stage.get_resolution()
            self.stages.append(stage)
        self.stages = nn.ModuleList(self.stages)

    def forward(self, hidden_state, output_hidden_states=False, return_dict=True):
        all_hidden_states = () if output_hidden_states else None
        for stage in self.stages:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)
            hidden_state = stage(hidden_state)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)
        if not return_dict:
            return tuple((v for v in [hidden_state, all_hidden_states] if v is not None))
        return BaseModelOutputWithNoAttention(last_hidden_state=hidden_state, hidden_states=all_hidden_states)
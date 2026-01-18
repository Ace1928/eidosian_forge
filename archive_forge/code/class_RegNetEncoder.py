from typing import Optional
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_regnet import RegNetConfig
class RegNetEncoder(nn.Module):

    def __init__(self, config: RegNetConfig):
        super().__init__()
        self.stages = nn.ModuleList([])
        self.stages.append(RegNetStage(config, config.embedding_size, config.hidden_sizes[0], stride=2 if config.downsample_in_first_stage else 1, depth=config.depths[0]))
        in_out_channels = zip(config.hidden_sizes, config.hidden_sizes[1:])
        for (in_channels, out_channels), depth in zip(in_out_channels, config.depths[1:]):
            self.stages.append(RegNetStage(config, in_channels, out_channels, depth=depth))

    def forward(self, hidden_state: Tensor, output_hidden_states: bool=False, return_dict: bool=True) -> BaseModelOutputWithNoAttention:
        hidden_states = () if output_hidden_states else None
        for stage_module in self.stages:
            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state,)
            hidden_state = stage_module(hidden_state)
        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)
        if not return_dict:
            return tuple((v for v in [hidden_state, hidden_states] if v is not None))
        return BaseModelOutputWithNoAttention(last_hidden_state=hidden_state, hidden_states=hidden_states)
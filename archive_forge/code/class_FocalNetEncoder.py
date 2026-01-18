import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BackboneOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_focalnet import FocalNetConfig
class FocalNetEncoder(nn.Module):

    def __init__(self, config, grid_size):
        super().__init__()
        self.num_stages = len(config.depths)
        self.config = config
        self.stages = nn.ModuleList([FocalNetStage(config=config, index=i_layer, input_resolution=(grid_size[0] // 2 ** i_layer, grid_size[1] // 2 ** i_layer)) for i_layer in range(self.num_stages)])
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor, input_dimensions: Tuple[int, int], output_hidden_states: Optional[bool]=False, output_hidden_states_before_downsampling: Optional[bool]=False, return_dict: Optional[bool]=True) -> Union[Tuple, FocalNetEncoderOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_reshaped_hidden_states = () if output_hidden_states else None
        if output_hidden_states:
            batch_size, _, hidden_size = hidden_states.shape
            reshaped_hidden_state = hidden_states.view(batch_size, *input_dimensions, hidden_size)
            reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
            all_hidden_states += (hidden_states,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)
        for i, stage_module in enumerate(self.stages):
            if self.gradient_checkpointing and self.training:
                stage_outputs = self._gradient_checkpointing_func(stage_module.__call__, hidden_states, input_dimensions)
            else:
                stage_outputs = stage_module(hidden_states, input_dimensions)
            hidden_states = stage_outputs[0]
            hidden_states_before_downsampling = stage_outputs[1]
            output_dimensions = stage_outputs[2]
            input_dimensions = (output_dimensions[-2], output_dimensions[-1])
            if output_hidden_states and output_hidden_states_before_downsampling:
                batch_size, _, hidden_size = hidden_states_before_downsampling.shape
                reshaped_hidden_state = hidden_states_before_downsampling.view(batch_size, *(output_dimensions[0], output_dimensions[1]), hidden_size)
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states_before_downsampling,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            elif output_hidden_states and (not output_hidden_states_before_downsampling):
                batch_size, _, hidden_size = hidden_states.shape
                reshaped_hidden_state = hidden_states.view(batch_size, *input_dimensions, hidden_size)
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states] if v is not None))
        return FocalNetEncoderOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, reshaped_hidden_states=all_reshaped_hidden_states)
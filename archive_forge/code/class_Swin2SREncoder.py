import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageSuperResolutionOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
from .configuration_swin2sr import Swin2SRConfig
class Swin2SREncoder(nn.Module):

    def __init__(self, config, grid_size):
        super().__init__()
        self.num_stages = len(config.depths)
        self.config = config
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        self.stages = nn.ModuleList([Swin2SRStage(config=config, dim=config.embed_dim, input_resolution=(grid_size[0], grid_size[1]), depth=config.depths[stage_idx], num_heads=config.num_heads[stage_idx], drop_path=dpr[sum(config.depths[:stage_idx]):sum(config.depths[:stage_idx + 1])], pretrained_window_size=0) for stage_idx in range(self.num_stages)])
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor, input_dimensions: Tuple[int, int], head_mask: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=False, output_hidden_states: Optional[bool]=False, return_dict: Optional[bool]=True) -> Union[Tuple, Swin2SREncoderOutput]:
        all_input_dimensions = ()
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        for i, stage_module in enumerate(self.stages):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(stage_module.__call__, hidden_states, input_dimensions, layer_head_mask, output_attentions)
            else:
                layer_outputs = stage_module(hidden_states, input_dimensions, layer_head_mask, output_attentions)
            hidden_states = layer_outputs[0]
            output_dimensions = layer_outputs[1]
            input_dimensions = (output_dimensions[-2], output_dimensions[-1])
            all_input_dimensions += (input_dimensions,)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if output_attentions:
                all_self_attentions += layer_outputs[2:]
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None))
        return Swin2SREncoderOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions)
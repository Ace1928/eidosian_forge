import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
from .configuration_donut_swin import DonutSwinConfig
class DonutSwinEncoder(nn.Module):

    def __init__(self, config, grid_size):
        super().__init__()
        self.num_layers = len(config.depths)
        self.config = config
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        self.layers = nn.ModuleList([DonutSwinStage(config=config, dim=int(config.embed_dim * 2 ** i_layer), input_resolution=(grid_size[0] // 2 ** i_layer, grid_size[1] // 2 ** i_layer), depth=config.depths[i_layer], num_heads=config.num_heads[i_layer], drop_path=dpr[sum(config.depths[:i_layer]):sum(config.depths[:i_layer + 1])], downsample=DonutSwinPatchMerging if i_layer < self.num_layers - 1 else None) for i_layer in range(self.num_layers)])
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor, input_dimensions: Tuple[int, int], head_mask: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=False, output_hidden_states: Optional[bool]=False, output_hidden_states_before_downsampling: Optional[bool]=False, always_partition: Optional[bool]=False, return_dict: Optional[bool]=True) -> Union[Tuple, DonutSwinEncoderOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_reshaped_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        if output_hidden_states:
            batch_size, _, hidden_size = hidden_states.shape
            reshaped_hidden_state = hidden_states.view(batch_size, *input_dimensions, hidden_size)
            reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
            all_hidden_states += (hidden_states,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)
        for i, layer_module in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(layer_module.__call__, hidden_states, input_dimensions, layer_head_mask, output_attentions, always_partition)
            else:
                layer_outputs = layer_module(hidden_states, input_dimensions, layer_head_mask, output_attentions, always_partition)
            hidden_states = layer_outputs[0]
            hidden_states_before_downsampling = layer_outputs[1]
            output_dimensions = layer_outputs[2]
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
            if output_attentions:
                all_self_attentions += layer_outputs[3:]
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None))
        return DonutSwinEncoderOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions, reshaped_hidden_states=all_reshaped_hidden_states)
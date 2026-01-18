import copy
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_canine import CanineConfig
class CanineEncoder(nn.Module):

    def __init__(self, config, local=False, always_attend_to_first_position=False, first_position_attends_to_all=False, attend_from_chunk_width=128, attend_from_chunk_stride=128, attend_to_chunk_width=128, attend_to_chunk_stride=128):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([CanineLayer(config, local, always_attend_to_first_position, first_position_attends_to_all, attend_from_chunk_width, attend_from_chunk_stride, attend_to_chunk_width, attend_to_chunk_stride) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self, hidden_states: Tuple[torch.FloatTensor], attention_mask: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=False, output_hidden_states: Optional[bool]=False, return_dict: Optional[bool]=True) -> Union[Tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(layer_module.__call__, hidden_states, attention_mask, layer_head_mask, output_attentions)
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions)
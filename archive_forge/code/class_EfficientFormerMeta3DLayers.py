import itertools
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_efficientformer import EfficientFormerConfig
class EfficientFormerMeta3DLayers(nn.Module):

    def __init__(self, config: EfficientFormerConfig):
        super().__init__()
        drop_paths = [config.drop_path_rate * (block_idx + sum(config.depths[:-1])) for block_idx in range(config.num_meta3d_blocks)]
        self.blocks = nn.ModuleList([EfficientFormerMeta3D(config, config.hidden_sizes[-1], drop_path=drop_path) for drop_path in drop_paths])

    def forward(self, hidden_states: torch.Tensor, output_attentions: bool=False) -> Tuple[torch.Tensor]:
        all_attention_outputs = () if output_attentions else None
        for layer_module in self.blocks:
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            hidden_states = layer_module(hidden_states, output_attentions)
            if output_attentions:
                all_attention_outputs = all_attention_outputs + (hidden_states[1],)
        if output_attentions:
            outputs = (hidden_states[0],) + all_attention_outputs
            return outputs
        return hidden_states
import collections.abc
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_groupvit import GroupViTConfig, GroupViTTextConfig, GroupViTVisionConfig
class GroupViTStage(nn.Module):
    """This corresponds to the `GroupingLayer` class in the GroupViT implementation."""

    def __init__(self, config: GroupViTVisionConfig, depth: int, num_prev_group_token: int, num_group_token: int, num_output_group: int):
        super().__init__()
        self.depth = depth
        self.num_group_token = num_group_token
        if num_group_token > 0:
            self.group_token = nn.Parameter(torch.zeros(1, num_group_token, config.hidden_size))
        else:
            self.group_token = None
        self.layers = nn.ModuleList([GroupViTEncoderLayer(config) for _ in range(depth)])
        if num_group_token > 0:
            self.downsample = GroupViTTokenAssign(config=config, num_group_token=num_group_token, num_output_group=num_output_group)
        else:
            self.downsample = None
        if num_prev_group_token > 0 and num_group_token > 0:
            self.group_projector = nn.Sequential(nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps), GroupViTMixerMLP(config, num_prev_group_token, config.hidden_size // 2, num_group_token))
        else:
            self.group_projector = None

    @property
    def with_group_token(self):
        return self.group_token is not None

    def split_x(self, x):
        if self.with_group_token:
            return (x[:, :-self.num_group_token], x[:, -self.num_group_token:])
        else:
            return (x, None)

    def concat_x(self, x: torch.Tensor, group_token: Optional[torch.Tensor]=None) -> torch.Tensor:
        if group_token is None:
            return x
        return torch.cat([x, group_token], dim=1)

    def forward(self, hidden_states: torch.Tensor, prev_group_token: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=False) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the grouping tensors of Grouping block.
        """
        if self.with_group_token:
            group_token = self.group_token.expand(hidden_states.size(0), -1, -1)
            if self.group_projector is not None:
                group_token = group_token + self.group_projector(prev_group_token)
        else:
            group_token = None
        x = hidden_states
        cat_x = self.concat_x(x, group_token)
        for layer in self.layers:
            layer_out = layer(cat_x, attention_mask=None, causal_attention_mask=None)
            cat_x = layer_out[0]
        x, group_token = self.split_x(cat_x)
        attention = None
        if self.downsample is not None:
            x, attention = self.downsample(x, group_token)
        outputs = (x, group_token)
        if output_attentions:
            outputs = outputs + (attention,)
        return outputs
from copy import copy
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_x_clip import XCLIPConfig, XCLIPTextConfig, XCLIPVisionConfig
class PromptGeneratorLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        embed_dim = config.projection_dim
        self.cross_attn = XCLIPCrossAttention(config)
        self.norm1 = nn.LayerNorm(embed_dim, eps=config.text_config.layer_norm_eps)
        self.norm3 = nn.LayerNorm(embed_dim, eps=config.text_config.layer_norm_eps)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), ACT2FN[config.prompt_hidden_act], nn.Dropout(config.prompt_attention_dropout), nn.Linear(embed_dim * 4, embed_dim))

    def forward(self, x, visual):
        x = x + self.cross_attn(self.norm1(x), visual, visual)
        x = x + self.mlp(self.norm3(x))
        return x
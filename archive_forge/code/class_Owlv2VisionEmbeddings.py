import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_owlv2 import Owlv2Config, Owlv2TextConfig, Owlv2VisionConfig
class Owlv2VisionEmbeddings(nn.Module):

    def __init__(self, config: Owlv2VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.class_embedding = nn.Parameter(torch.randn(config.hidden_size))
        self.patch_embedding = nn.Conv2d(in_channels=config.num_channels, out_channels=self.embed_dim, kernel_size=config.patch_size, stride=config.patch_size, bias=False)
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer('position_ids', torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings
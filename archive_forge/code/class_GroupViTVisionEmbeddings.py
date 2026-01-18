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
class GroupViTVisionEmbeddings(nn.Module):

    def __init__(self, config: GroupViTVisionConfig):
        super().__init__()
        self.patch_embeddings = GroupViTPatchEmbeddings(image_size=config.image_size, patch_size=config.patch_size, num_channels=config.num_channels, embed_dim=config.hidden_size)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches, config.hidden_size))
        self.dropout = nn.Dropout(config.dropout)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.config = config

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """
        npatch = embeddings.shape[1]
        if npatch == self.position_embeddings.shape[1] and height == width:
            return self.position_embeddings
        patch_pos_embed = self.position_embeddings
        num_original_pos_embed = patch_pos_embed.shape[1]
        dim = embeddings.shape[-1]
        feat_height = height // self.config.patch_size
        feat_width = width // self.config.patch_size
        feat_height, feat_width = (feat_height + 0.1, feat_width + 0.1)
        original_height = original_width = math.sqrt(num_original_pos_embed)
        reshaped_patch_pos_embed = patch_pos_embed.reshape(1, int(original_height), int(original_width), dim).permute(0, 3, 1, 2)
        scale_factor = (feat_height / original_height, feat_width / original_width)
        patch_pos_embed = nn.functional.interpolate(reshaped_patch_pos_embed, scale_factor=scale_factor, mode='bicubic', align_corners=False)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool=False) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        embeddings = self.layernorm(embeddings)
        batch_size, seq_len, _ = embeddings.size()
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings
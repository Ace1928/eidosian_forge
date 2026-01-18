import collections
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_flava import (
class FlavaImageEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config: FlavaImageConfig, use_mask_token: bool=False) -> None:
        super().__init__()
        use_mask_token = use_mask_token or config.mask_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
        self.patch_embeddings = PatchEmbeddings(image_size=config.image_size, patch_size=config.patch_size, num_channels=config.num_channels, embed_dim=config.hidden_size)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/image_transformer.py#L174
        """
        npatch = embeddings.shape[1] - 1
        num_pos = self.position_embeddings.shape[1] - 1
        if npatch == num_pos and height == width:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        num_h_patches = height // self.config.patch_size
        num_w_patches = width // self.config.patch_size
        num_h_patches, num_w_patches = (num_h_patches + 0.1, num_w_patches + 0.1)
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed.reshape(1, int(math.sqrt(num_pos)), int(math.sqrt(num_pos)), dim).permute(0, 3, 1, 2), scale_factor=(num_h_patches / math.sqrt(num_pos), num_w_patches / math.sqrt(num_pos)), mode='bicubic', align_corners=False)
        if int(num_h_patches) != patch_pos_embed.shape[-2] or int(num_w_patches) != patch_pos_embed.shape[-1]:
            raise ValueError(f"Number of patches for images ({(int(num_h_patches), int(num_w_patches))}) don't match the shape of position embedding ({(patch_pos_embed.shape[-2], patch_pos_embed.shape[-1])})")
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: Optional[torch.BoolTensor]=None, interpolate_pos_encoding: bool=False) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        batch_size, seq_len, _ = embeddings.size()
        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            if bool_masked_pos.dim() == 3:
                bool_masked_pos = bool_masked_pos.view(bool_masked_pos.size(0), -1)
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings
import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_beit import BeitConfig
class BeitEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.

    """

    def __init__(self, config: BeitConfig) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        if config.use_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        else:
            self.mask_token = None
        self.patch_embeddings = BeitPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        if config.use_absolute_position_embeddings:
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        else:
            self.position_embeddings = None
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: Optional[torch.BoolTensor]=None) -> torch.Tensor:
        embeddings, (patch_height, patch_width) = self.patch_embeddings(pixel_values, self.position_embeddings[:, 1:, :] if self.position_embeddings is not None else None)
        batch_size, seq_len, _ = embeddings.size()
        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            w = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1 - w) + mask_tokens * w
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        if self.position_embeddings is not None:
            cls_tokens = cls_tokens + self.position_embeddings[:, :1, :]
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = self.dropout(embeddings)
        return (embeddings, (patch_height, patch_width))
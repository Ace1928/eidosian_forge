import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...file_utils import (
from ...modeling_outputs import BaseModelOutput, DepthEstimatorOutput, SemanticSegmenterOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import ModelOutput, logging
from ...utils.backbone_utils import load_backbone
from .configuration_dpt import DPTConfig
class DPTViTEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.

    """

    def __init__(self, config):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = DPTViTPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def _resize_pos_embed(self, posemb, grid_size_height, grid_size_width, start_index=1):
        posemb_tok = posemb[:, :start_index]
        posemb_grid = posemb[0, start_index:]
        old_grid_size = int(math.sqrt(len(posemb_grid)))
        posemb_grid = posemb_grid.reshape(1, old_grid_size, old_grid_size, -1).permute(0, 3, 1, 2)
        posemb_grid = nn.functional.interpolate(posemb_grid, size=(grid_size_height, grid_size_width), mode='bilinear')
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, grid_size_height * grid_size_width, -1)
        posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
        return posemb

    def forward(self, pixel_values, return_dict=False):
        batch_size, num_channels, height, width = pixel_values.shape
        patch_size = self.config.patch_size
        position_embeddings = self._resize_pos_embed(self.position_embeddings, height // patch_size, width // patch_size)
        embeddings = self.patch_embeddings(pixel_values)
        batch_size, seq_len, _ = embeddings.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = embeddings + position_embeddings
        embeddings = self.dropout(embeddings)
        if not return_dict:
            return (embeddings,)
        return BaseModelOutputWithIntermediateActivations(last_hidden_states=embeddings)
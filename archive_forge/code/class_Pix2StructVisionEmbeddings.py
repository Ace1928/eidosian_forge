import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_pix2struct import Pix2StructConfig, Pix2StructTextConfig, Pix2StructVisionConfig
class Pix2StructVisionEmbeddings(nn.Module):
    """
    Construct the embeddings from patch. In `Pix2Struct` the input is different from classic Vision-transformer models.
    Here the input is a sequence of `seq_len` flattened patches that also combines padding patches (tokens). Each patch
    is represented by a vector of `hidden_size` values.
    """

    def __init__(self, config: Pix2StructConfig) -> None:
        super().__init__()
        self.patch_projection = nn.Linear(config.patch_embed_hidden_size, config.hidden_size)
        self.row_embedder = nn.Embedding(config.seq_len, config.hidden_size)
        self.column_embedder = nn.Embedding(config.seq_len, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, flattened_patches: torch.Tensor) -> torch.Tensor:
        row_indices = flattened_patches[:, :, 0].long()
        col_indices = flattened_patches[:, :, 1].long()
        flattened_patches = flattened_patches[:, :, 2:]
        embeddings = self.patch_projection(flattened_patches)
        row_embeddings = self.row_embedder(row_indices)
        col_embeddings = self.column_embedder(col_indices)
        embeddings = embeddings + row_embeddings + col_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings
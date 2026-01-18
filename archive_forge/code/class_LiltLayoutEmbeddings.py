import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_lilt import LiltConfig
class LiltLayoutEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size // 6)
        self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size // 6)
        self.h_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size // 6)
        self.w_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size // 6)
        self.padding_idx = config.pad_token_id
        self.box_position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size // config.channel_shrink_ratio, padding_idx=self.padding_idx)
        self.box_linear_embeddings = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size // config.channel_shrink_ratio)
        self.LayerNorm = nn.LayerNorm(config.hidden_size // config.channel_shrink_ratio, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, bbox=None, position_ids=None):
        try:
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError('The `bbox` coordinate values should be within 0-1000 range.') from e
        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])
        spatial_position_embeddings = torch.cat([left_position_embeddings, upper_position_embeddings, right_position_embeddings, lower_position_embeddings, h_position_embeddings, w_position_embeddings], dim=-1)
        spatial_position_embeddings = self.box_linear_embeddings(spatial_position_embeddings)
        box_position_embeddings = self.box_position_embeddings(position_ids)
        spatial_position_embeddings = spatial_position_embeddings + box_position_embeddings
        spatial_position_embeddings = self.LayerNorm(spatial_position_embeddings)
        spatial_position_embeddings = self.dropout(spatial_position_embeddings)
        return spatial_position_embeddings
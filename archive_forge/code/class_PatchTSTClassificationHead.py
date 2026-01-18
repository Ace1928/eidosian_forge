import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from ...activations import ACT2CLS
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...time_series_utils import NegativeBinomialOutput, NormalOutput, StudentTOutput
from ...utils import ModelOutput, add_start_docstrings, logging
from .configuration_patchtst import PatchTSTConfig
class PatchTSTClassificationHead(nn.Module):

    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.use_cls_token = config.use_cls_token
        self.pooling_type = config.pooling_type
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity()
        self.linear = nn.Linear(config.num_input_channels * config.d_model, config.num_targets)

    def forward(self, embedding: torch.Tensor):
        """
        Parameters:
            embedding (`torch.Tensor` of shape `(bs, num_channels, num_patches, d_model)` or
                     `(bs, num_channels, num_patches+1, d_model)` if `cls_token` is set to True, *required*):
                Embedding from the model
        Returns:
            `torch.Tensor` of shape `(bs, num_targets)`

        """
        if self.use_cls_token:
            pooled_embedding = embedding[:, :, 0, :]
        elif self.pooling_type == 'mean':
            pooled_embedding = embedding.mean(dim=2)
        elif self.pooling_type == 'max':
            pooled_embedding = embedding.max(dim=2).values
        else:
            raise ValueError(f'pooling operator {self.pooling_type} is not implemented yet')
        pooled_embedding = self.flatten(pooled_embedding)
        output = self.linear(self.dropout(pooled_embedding))
        return output
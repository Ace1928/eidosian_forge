import copy
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from .configuration_oneformer import OneFormerConfig
class OneFormerTransformerModule(nn.Module):
    """
    The OneFormer's transformer module.
    """

    def __init__(self, in_features: int, config: OneFormerConfig):
        super().__init__()
        hidden_dim = config.hidden_dim
        self.num_feature_levels = 3
        self.position_embedder = OneFormerSinePositionEmbedding(num_pos_feats=hidden_dim // 2, normalize=True)
        self.queries_embedder = nn.Embedding(config.num_queries, hidden_dim)
        self.input_projections = []
        for _ in range(self.num_feature_levels):
            if in_features != hidden_dim or config.enforce_input_proj:
                self.input_projections.append(nn.Conv2d(in_features, hidden_dim, kernel_size=1))
            else:
                self.input_projections.append(nn.Sequential())
        self.decoder = OneFormerTransformerDecoder(in_channels=in_features, config=config)
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

    def forward(self, multi_scale_features: List[Tensor], mask_features: Tensor, task_token: Tensor, output_attentions: bool=False) -> OneFormerTransformerDecoderOutput:
        if not len(multi_scale_features) == self.num_feature_levels:
            raise ValueError(f'Number of elements in multi_scale_features ({len(multi_scale_features)}) and num_feature_levels ({self.num_feature_levels}) do not match!')
        multi_stage_features = []
        multi_stage_positional_embeddings = []
        size_list = []
        for i in range(self.num_feature_levels):
            size_list.append(multi_scale_features[i].shape[-2:])
            multi_stage_positional_embeddings.append(self.position_embedder(multi_scale_features[i], None).flatten(2))
            multi_stage_features.append(self.input_projections[i](multi_scale_features[i]).flatten(2) + self.level_embed.weight[i][None, :, None])
            multi_stage_positional_embeddings[-1] = multi_stage_positional_embeddings[-1].permute(2, 0, 1)
            multi_stage_features[-1] = multi_stage_features[-1].permute(2, 0, 1)
        _, batch_size, _ = multi_stage_features[0].shape
        query_embeddings = self.queries_embedder.weight.unsqueeze(1).repeat(1, batch_size, 1)
        task_token = task_token.unsqueeze(0)
        query_features = self.position_embedder(mask_features, None)
        return self.decoder(task_token=task_token, multi_stage_features=multi_stage_features, multi_stage_positional_embeddings=multi_stage_positional_embeddings, mask_features=mask_features, query_features=query_features, query_embeddings=query_embeddings, query_embedder=self.queries_embedder, size_list=size_list, output_attentions=output_attentions)
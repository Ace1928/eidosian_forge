import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_bros import BrosConfig
class BrosRelationExtractor(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_relations = config.n_relations
        self.backbone_hidden_size = config.hidden_size
        self.head_hidden_size = config.hidden_size
        self.classifier_dropout_prob = config.classifier_dropout_prob
        self.drop = nn.Dropout(self.classifier_dropout_prob)
        self.query = nn.Linear(self.backbone_hidden_size, self.n_relations * self.head_hidden_size)
        self.key = nn.Linear(self.backbone_hidden_size, self.n_relations * self.head_hidden_size)
        self.dummy_node = nn.Parameter(torch.zeros(1, self.backbone_hidden_size))

    def forward(self, query_layer: torch.Tensor, key_layer: torch.Tensor):
        query_layer = self.query(self.drop(query_layer))
        dummy_vec = self.dummy_node.unsqueeze(0).repeat(1, key_layer.size(1), 1)
        key_layer = torch.cat([key_layer, dummy_vec], axis=0)
        key_layer = self.key(self.drop(key_layer))
        query_layer = query_layer.view(query_layer.size(0), query_layer.size(1), self.n_relations, self.head_hidden_size)
        key_layer = key_layer.view(key_layer.size(0), key_layer.size(1), self.n_relations, self.head_hidden_size)
        relation_score = torch.matmul(query_layer.permute(2, 1, 0, 3), key_layer.permute(2, 1, 3, 0))
        return relation_score
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_realm import RealmConfig
class RealmScorerProjection(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.predictions = RealmLMPredictionHead(config)
        self.dense = nn.Linear(config.hidden_size, config.retriever_proj_size)
        self.LayerNorm = nn.LayerNorm(config.retriever_proj_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states
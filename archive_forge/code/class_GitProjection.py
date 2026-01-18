import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...file_utils import ModelOutput
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_git import GitConfig, GitVisionConfig
class GitProjection(nn.Module):

    def __init__(self, config: GitConfig):
        super().__init__()
        self.config = config
        self.visual_projection = nn.Sequential(nn.Linear(config.vision_config.hidden_size, config.hidden_size), nn.LayerNorm(config.hidden_size, eps=config.vision_config.layer_norm_eps))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.visual_projection(embeddings)
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
@dataclass
class RealmScorerOutput(ModelOutput):
    """
    Outputs of [`RealmScorer`] models.

    Args:
        relevance_score (`torch.FloatTensor` of shape `(batch_size, config.num_candidates)`):
            The relevance score of document candidates (before softmax).
        query_score (`torch.FloatTensor` of shape `(batch_size, config.retriever_proj_size)`):
            Query score derived from the query embedder.
        candidate_score (`torch.FloatTensor` of shape `(batch_size, config.num_candidates, config.retriever_proj_size)`):
            Candidate score derived from the embedder.
    """
    relevance_score: torch.FloatTensor = None
    query_score: torch.FloatTensor = None
    candidate_score: torch.FloatTensor = None
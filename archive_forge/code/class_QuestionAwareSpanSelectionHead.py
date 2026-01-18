import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, ModelOutput, QuestionAnsweringModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_splinter import SplinterConfig
class QuestionAwareSpanSelectionHead(nn.Module):
    """
    Implementation of Question-Aware Span Selection (QASS) head, described in Splinter's paper:

    """

    def __init__(self, config):
        super().__init__()
        self.query_start_transform = SplinterFullyConnectedLayer(config.hidden_size, config.hidden_size)
        self.query_end_transform = SplinterFullyConnectedLayer(config.hidden_size, config.hidden_size)
        self.start_transform = SplinterFullyConnectedLayer(config.hidden_size, config.hidden_size)
        self.end_transform = SplinterFullyConnectedLayer(config.hidden_size, config.hidden_size)
        self.start_classifier = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.end_classifier = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, inputs, positions):
        _, _, dim = inputs.size()
        index = positions.unsqueeze(-1).repeat(1, 1, dim)
        gathered_reps = torch.gather(inputs, dim=1, index=index)
        query_start_reps = self.query_start_transform(gathered_reps)
        query_end_reps = self.query_end_transform(gathered_reps)
        start_reps = self.start_transform(inputs)
        end_reps = self.end_transform(inputs)
        hidden_states = self.start_classifier(query_start_reps)
        start_reps = start_reps.permute(0, 2, 1)
        start_logits = torch.matmul(hidden_states, start_reps)
        hidden_states = self.end_classifier(query_end_reps)
        end_reps = end_reps.permute(0, 2, 1)
        end_logits = torch.matmul(hidden_states, end_reps)
        return (start_logits, end_logits)
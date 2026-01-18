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
class RealmReaderProjection(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense_intermediate = nn.Linear(config.hidden_size, config.span_hidden_size * 2)
        self.dense_output = nn.Linear(config.span_hidden_size, 1)
        self.layer_normalization = nn.LayerNorm(config.span_hidden_size, eps=config.reader_layer_norm_eps)
        self.relu = nn.ReLU()

    def forward(self, hidden_states, block_mask):

        def span_candidates(masks):
            """
            Generate span candidates.

            Args:
                masks: <bool> [num_retrievals, max_sequence_len]

            Returns:
                starts: <int32> [num_spans] ends: <int32> [num_spans] span_masks: <int32> [num_retrievals, num_spans]
                whether spans locate in evidence block.
            """
            _, max_sequence_len = masks.shape

            def _spans_given_width(width):
                current_starts = torch.arange(max_sequence_len - width + 1, device=masks.device)
                current_ends = torch.arange(width - 1, max_sequence_len, device=masks.device)
                return (current_starts, current_ends)
            starts, ends = zip(*(_spans_given_width(w + 1) for w in range(self.config.max_span_width)))
            starts = torch.cat(starts, 0)
            ends = torch.cat(ends, 0)
            start_masks = torch.index_select(masks, dim=-1, index=starts)
            end_masks = torch.index_select(masks, dim=-1, index=ends)
            span_masks = start_masks * end_masks
            return (starts, ends, span_masks)

        def mask_to_score(mask, dtype=torch.float32):
            return (1.0 - mask.type(dtype)) * torch.finfo(dtype).min
        hidden_states = self.dense_intermediate(hidden_states)
        start_projection, end_projection = hidden_states.chunk(2, dim=-1)
        candidate_starts, candidate_ends, candidate_mask = span_candidates(block_mask)
        candidate_start_projections = torch.index_select(start_projection, dim=1, index=candidate_starts)
        candidate_end_projections = torch.index_select(end_projection, dim=1, index=candidate_ends)
        candidate_hidden = candidate_start_projections + candidate_end_projections
        candidate_hidden = self.relu(candidate_hidden)
        candidate_hidden = self.layer_normalization(candidate_hidden)
        reader_logits = self.dense_output(candidate_hidden).squeeze(-1)
        reader_logits += mask_to_score(candidate_mask, dtype=reader_logits.dtype)
        return (reader_logits, candidate_starts, candidate_ends)
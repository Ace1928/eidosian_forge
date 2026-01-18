from collections.abc import Sequence
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import softmax_backward_data
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_deberta import DebertaConfig
class DebertaEncoder(nn.Module):
    """Modified BertEncoder with relative position bias support"""

    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([DebertaLayer(config) for _ in range(config.num_hidden_layers)])
        self.relative_attention = getattr(config, 'relative_attention', False)
        if self.relative_attention:
            self.max_relative_positions = getattr(config, 'max_relative_positions', -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.rel_embeddings = nn.Embedding(self.max_relative_positions * 2, config.hidden_size)
        self.gradient_checkpointing = False

    def get_rel_embedding(self):
        rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
        return rel_embeddings

    def get_attention_mask(self, attention_mask):
        if attention_mask.dim() <= 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)
        return attention_mask

    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        if self.relative_attention and relative_pos is None:
            q = query_states.size(-2) if query_states is not None else hidden_states.size(-2)
            relative_pos = build_relative_position(q, hidden_states.size(-2), hidden_states.device)
        return relative_pos

    def forward(self, hidden_states, attention_mask, output_hidden_states=True, output_attentions=False, query_states=None, relative_pos=None, return_dict=True):
        attention_mask = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        if isinstance(hidden_states, Sequence):
            next_kv = hidden_states[0]
        else:
            next_kv = hidden_states
        rel_embeddings = self.get_rel_embedding()
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(layer_module.__call__, next_kv, attention_mask, query_states, relative_pos, rel_embeddings, output_attentions)
            else:
                hidden_states = layer_module(next_kv, attention_mask, query_states=query_states, relative_pos=relative_pos, rel_embeddings=rel_embeddings, output_attentions=output_attentions)
            if output_attentions:
                hidden_states, att_m = hidden_states
            if query_states is not None:
                query_states = hidden_states
                if isinstance(hidden_states, Sequence):
                    next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
            else:
                next_kv = hidden_states
            if output_attentions:
                all_attentions = all_attentions + (att_m,)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions)
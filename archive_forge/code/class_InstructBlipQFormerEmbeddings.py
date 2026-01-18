import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from ..auto import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from .configuration_instructblip import InstructBlipConfig, InstructBlipQFormerConfig, InstructBlipVisionConfig
class InstructBlipQFormerEmbeddings(nn.Module):
    """Construct the embeddings from word and position embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer('position_ids', torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False)
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'absolute')
        self.config = config

    def forward(self, input_ids=None, position_ids=None, query_embeds=None, past_key_values_length=0):
        if input_ids is not None:
            seq_length = input_ids.size()[1]
        else:
            seq_length = 0
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length:seq_length + past_key_values_length].clone()
        if input_ids is not None:
            embeddings = self.word_embeddings(input_ids)
            if self.position_embedding_type == 'absolute':
                position_embeddings = self.position_embeddings(position_ids.to(embeddings.device))
                embeddings = embeddings + position_embeddings
            if query_embeds is not None:
                embeddings = torch.cat((query_embeds, embeddings), dim=1)
        else:
            embeddings = query_embeds
        embeddings = embeddings.to(self.layernorm.weight.dtype)
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
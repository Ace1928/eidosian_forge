import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_funnel import FunnelConfig
class FunnelEmbeddings(nn.Module):

    def __init__(self, config: FunnelConfig) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, input_ids: Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        embeddings = self.layer_norm(inputs_embeds)
        embeddings = self.dropout(embeddings)
        return embeddings
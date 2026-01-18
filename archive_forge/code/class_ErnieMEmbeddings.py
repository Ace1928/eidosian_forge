import math
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn, tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_ernie_m import ErnieMConfig
class ErnieMEmbeddings(nn.Module):
    """Construct the embeddings from word and position embeddings."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=config.pad_token_id)
        self.layer_norm = nn.LayerNorm(normalized_shape=config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.padding_idx = config.pad_token_id

    def forward(self, input_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, inputs_embeds: Optional[torch.LongTensor]=None, past_key_values_length: int=0) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        if position_ids is None:
            input_shape = inputs_embeds.size()[:-1]
            ones = torch.ones(input_shape, dtype=torch.int64, device=inputs_embeds.device)
            seq_length = torch.cumsum(ones, dim=1)
            position_ids = seq_length - ones
            if past_key_values_length > 0:
                position_ids = position_ids + past_key_values_length
        position_ids += 2
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
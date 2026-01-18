import enum
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, MaskedLMOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import (
from ...utils import (
from .configuration_tapas import TapasConfig
class TapasEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings. Same as BertEmbeddings but with a number of
    additional token type embeddings to encode tabular structure.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        for i, type_vocab_sizes in enumerate(config.type_vocab_sizes):
            name = f'token_type_embeddings_{i}'
            setattr(self, name, nn.Embedding(type_vocab_sizes, config.hidden_size))
        self.number_of_token_type_embeddings = len(config.type_vocab_sizes)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
            if self.config.reset_position_index_per_cell:
                col_index = IndexMap(token_type_ids[:, :, 1], self.config.type_vocab_sizes[1], batch_dims=1)
                row_index = IndexMap(token_type_ids[:, :, 2], self.config.type_vocab_sizes[2], batch_dims=1)
                full_index = ProductIndexMap(col_index, row_index)
                first_position_per_segment = reduce_min(position_ids, full_index)[0]
                first_position = gather(first_position_per_segment, full_index)
                position = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0)
                position_ids = torch.min(torch.as_tensor(self.config.max_position_embeddings - 1, device=device), position - first_position)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape + self.number_of_token_type_embeddings, dtype=torch.long, device=device)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings
        for i in range(self.number_of_token_type_embeddings):
            name = f'token_type_embeddings_{i}'
            embeddings += getattr(self, name)(token_type_ids[:, :, i])
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
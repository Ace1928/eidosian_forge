import sys
from collections import namedtuple
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.autograd.function import Function
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import CausalLMOutput, MaskedLMOutput, QuestionAnsweringModelOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_reformer import ReformerConfig
class ReformerEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.max_position_embeddings = config.max_position_embeddings
        self.dropout = config.hidden_dropout_prob
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = AxialPositionEmbeddings(config) if config.axial_pos_embds else PositionEmbeddings(config)

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None, start_idx_pos_encodings=0):
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        else:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = torch.arange(start_idx_pos_encodings, start_idx_pos_encodings + seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        if position_ids.shape[-1] > self.max_position_embeddings:
            raise ValueError(f'Sequence Length: {position_ids.shape[-1]} has to be less or equal than config.max_position_embeddings {self.max_position_embeddings}.')
        embeddings = nn.functional.dropout(inputs_embeds, p=self.dropout, training=self.training)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = embeddings + position_embeddings
        return embeddings
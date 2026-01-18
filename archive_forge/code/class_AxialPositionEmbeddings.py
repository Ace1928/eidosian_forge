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
class AxialPositionEmbeddings(nn.Module):
    """
    Constructs axial position embeddings. Useful for very long input sequences to save memory and time.
    """

    def __init__(self, config):
        super().__init__()
        self.axial_pos_shape = config.axial_pos_shape
        self.axial_pos_embds_dim = config.axial_pos_embds_dim
        self.dropout = config.hidden_dropout_prob
        self.least_common_mult_chunk_length = _get_least_common_mult_chunk_len(config)
        self.weights = nn.ParameterList()
        if sum(self.axial_pos_embds_dim) != config.hidden_size:
            raise ValueError(f'Make sure that config.axial_pos_embds factors: {self.axial_pos_embds_dim} sum to config.hidden_size: {config.hidden_size}')
        for axis, axial_pos_embd_dim in enumerate(self.axial_pos_embds_dim):
            ax_shape = [1] * len(self.axial_pos_shape)
            ax_shape[axis] = self.axial_pos_shape[axis]
            ax_shape = tuple(ax_shape) + (axial_pos_embd_dim,)
            self.weights.append(nn.Parameter(torch.ones(ax_shape, dtype=torch.float32)))

    def forward(self, position_ids):
        batch_size = position_ids.shape[0]
        sequence_length = position_ids.shape[1]
        broadcasted_weights = [weight.expand((batch_size,) + self.axial_pos_shape + weight.shape[-1:]) for weight in self.weights]
        if self.training is True:
            if reduce(mul, self.axial_pos_shape) != sequence_length:
                raise ValueError(f'If training, make sure that config.axial_pos_shape factors: {self.axial_pos_shape} multiply to sequence length. Got prod({self.axial_pos_shape}) != sequence_length: {sequence_length}. You might want to consider padding your sequence length to {reduce(mul, self.axial_pos_shape)} or changing config.axial_pos_shape.')
            if self.dropout > 0:
                weights = torch.cat(broadcasted_weights, dim=-1)
                transposed_weights = weights.transpose(2, 1)
                dropped_transposed_weights = nn.functional.dropout2d(transposed_weights, p=self.dropout, training=self.training)
                dropped_weights = dropped_transposed_weights.transpose(2, 1)
                position_encodings = torch.reshape(dropped_weights, (batch_size, sequence_length, -1))
            else:
                position_encodings = torch.cat([torch.reshape(weight, (batch_size, sequence_length, -1)) for weight in broadcasted_weights], dim=-1)
        else:
            if reduce(mul, self.axial_pos_shape) < sequence_length:
                raise ValueError(f'Make sure that config.axial_pos_shape factors: {self.axial_pos_shape} multiply at least to max(sequence_length, least_common_mult_chunk_length): max({sequence_length}, {self.least_common_mult_chunk_length}).')
            max_position_id = position_ids.max().item()
            required_pos_encodings_columns = -(-(max_position_id + 1) // self.axial_pos_shape[1])
            position_encodings = torch.cat([weight[:, :required_pos_encodings_columns] for weight in broadcasted_weights], dim=-1)
            position_encodings = torch.reshape(position_encodings, (batch_size, -1, position_encodings.shape[-1]))
            position_encodings = torch.cat([torch.index_select(position_encodings[i], 0, position_ids[i]).unsqueeze(0) for i in range(batch_size)], dim=0)
        return position_encodings
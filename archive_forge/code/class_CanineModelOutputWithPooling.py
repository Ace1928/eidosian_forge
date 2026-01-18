import copy
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_canine import CanineConfig
@dataclass
class CanineModelOutputWithPooling(ModelOutput):
    """
    Output type of [`CanineModel`]. Based on [`~modeling_outputs.BaseModelOutputWithPooling`], but with slightly
    different `hidden_states` and `attentions`, as these also include the hidden states and attentions of the shallow
    Transformer encoders.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model (i.e. the output of the final
            shallow Transformer encoder).
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Hidden-state of the first token of the sequence (classification token) at the last layer of the deep
            Transformer encoder, further processed by a Linear layer and a Tanh activation function. The Linear layer
            weights are trained from the next sentence prediction (classification) objective during pretraining.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the input to each encoder + one for the output of each layer of each
            encoder) of shape `(batch_size, sequence_length, hidden_size)` and `(batch_size, sequence_length //
            config.downsampling_rate, hidden_size)`. Hidden-states of the model at the output of each layer plus the
            initial input to each Transformer encoder. The hidden states of the shallow encoders have length
            `sequence_length`, but the hidden states of the deep encoder have length `sequence_length` //
            `config.downsampling_rate`.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of the 3 Transformer encoders of shape `(batch_size,
            num_heads, sequence_length, sequence_length)` and `(batch_size, num_heads, sequence_length //
            config.downsampling_rate, sequence_length // config.downsampling_rate)`. Attentions weights after the
            attention softmax, used to compute the weighted average in the self-attention heads.
    """
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
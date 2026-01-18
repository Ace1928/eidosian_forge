import collections.abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_tvlt import TvltConfig
@dataclass
class TvltModelOutput(ModelOutput):
    """
    Class for TvltModel's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        last_pixel_hidden_state (`torch.FloatTensor` of shape `(batch_size, pixel_sequence_length, hidden_size)`):
            Pixel sequence of hidden-states at the output of the last layer of the model.
        last_audio_hidden_state (`torch.FloatTensor` of shape `(batch_size, audio_sequence_length, hidden_size)`):
            Audio sequence of hidden-states at the output of the last layer of the model.
        pixel_label_masks (`torch.FloatTensor` of shape `(batch_size, pixel_patch_length)`):
            Tensor indicating which pixel patches are masked (1) and which are not (0).
        audio_label_masks (`torch.FloatTensor` of shape `(batch_size, audio_patch_length)`):
            Tensor indicating which audio patches are masked (1) and which are not (0).
        pixel_ids_restore (`torch.LongTensor` of shape `(batch_size, pixel_patch_length)`):
            Tensor containing the ids permutation of pixel masking.
        audio_ids_restore (`torch.LongTensor` of shape `(batch_size, audio_patch_length)`):
            Tensor containing the ids permutation of audio masking.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """
    last_hidden_state: torch.FloatTensor = None
    last_pixel_hidden_state: torch.FloatTensor = None
    last_audio_hidden_state: torch.FloatTensor = None
    pixel_label_masks: torch.LongTensor = None
    audio_label_masks: torch.LongTensor = None
    pixel_ids_restore: torch.LongTensor = None
    audio_ids_restore: torch.LongTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
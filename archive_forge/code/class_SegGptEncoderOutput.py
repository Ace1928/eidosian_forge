import collections.abc
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_seggpt import SegGptConfig
from ..deprecated._archive_maps import SEGGPT_PRETRAINED_MODEL_ARCHIVE_LIST  # noqa: F401, E402
@dataclass
class SegGptEncoderOutput(ModelOutput):
    """
    Output type of [`SegGptEncoderOutput`].
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, patch_height, patch_width, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`Tuple[torch.FloatTensor]`, `optional`, returned when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape `(batch_size, patch_height, patch_width, hidden_size)`.
        attentions (`Tuple[torch.FloatTensor]`, `optional`, returned when `config.output_attentions=True`):
            Tuple of *torch.FloatTensor* (one for each layer) of shape
            `(batch_size, num_heads, seq_len, seq_len)`.
        intermediate_hidden_states (`Tuple[torch.FloatTensor]`, `optional`, returned when `config.intermediate_hidden_state_indices` is set):
            Tuple of `torch.FloatTensor` of shape `(batch_size, patch_height, patch_width, hidden_size)`.
            Each element in the Tuple corresponds to the output of the layer specified in `config.intermediate_hidden_state_indices`.
            Additionaly, each feature passes through a LayerNorm.
    """
    last_hidden_state: torch.FloatTensor
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    intermediate_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
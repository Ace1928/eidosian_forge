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
class SegGptImageSegmentationOutput(ModelOutput):
    """
    Output type of [`SegGptImageSegmentationOutput`].

    Args:
        loss (`torch.FloatTensor`, `optional`, returned when `labels` is provided):
            The loss value.
        pred_masks (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The predicted masks.
        hidden_states (`Tuple[torch.FloatTensor]`, `optional`, returned when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape `(batch_size, patch_height, patch_width, hidden_size)`.
        attentions (`Tuple[torch.FloatTensor]`, `optional`, returned when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape
            `(batch_size, num_heads, seq_len, seq_len)`.
    """
    loss: Optional[torch.FloatTensor] = None
    pred_masks: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
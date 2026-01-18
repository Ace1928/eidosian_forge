import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.import_utils import is_causal_conv1d_available, is_mamba_ssm_available
from .configuration_mamba import MambaConfig
from ..deprecated._archive_maps import MAMBA_PRETRAINED_MODEL_ARCHIVE_LIST  # noqa: F401, E402
@dataclass
class MambaOutput(ModelOutput):
    """
    Class for the MAMBA model outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        cache_params (`MambaCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """
    last_hidden_state: Optional[torch.FloatTensor] = None
    cache_params: Optional[MambaCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
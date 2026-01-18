import collections
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_timesformer import TimesformerConfig
class TimesformerSelfOutput(nn.Module):
    """
    The residual connection is defined in TimesformerLayer instead of here (as is the case with other models), due to
    the layernorm applied before each block.
    """

    def __init__(self, config: TimesformerConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states
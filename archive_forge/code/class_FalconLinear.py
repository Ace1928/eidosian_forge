import math
import warnings
from typing import TYPE_CHECKING, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from torch.nn import functional as F
from ...modeling_attn_mask_utils import (
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import is_torch_greater_or_equal_than_2_0
from ...utils import (
from .configuration_falcon import FalconConfig
class FalconLinear(nn.Linear):

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden_states = input @ self.weight.T
        if self.bias is None:
            return hidden_states
        return hidden_states + self.bias
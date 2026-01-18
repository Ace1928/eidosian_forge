import math
import sys
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from ...integrations.deepspeed import is_deepspeed_available
from ...modeling_outputs import ModelOutput
from ...utils import (
from .configuration_esm import EsmConfig
from .modeling_esm import ESM_START_DOCSTRING, EsmModel, EsmPreTrainedModel
from .openfold_utils import (
class EsmFoldBackboneUpdate(nn.Module):
    """
    Implements part of Algorithm 23.
    """

    def __init__(self, config):
        super().__init__()
        self.linear = EsmFoldLinear(config.sequence_dim, 6, init='final')

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            [*, N_res, C_s] single representation
        Returns:
            [*, N_res, 6] update vector
        """
        update = self.linear(s)
        return update
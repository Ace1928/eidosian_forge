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
class EsmFoldLayerNorm(nn.Module):

    def __init__(self, c_in, eps=1e-05):
        super().__init__()
        self.c_in = (c_in,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(c_in))
        self.bias = nn.Parameter(torch.zeros(c_in))

    def forward(self, x):
        d = x.dtype
        if d is torch.bfloat16 and (not is_deepspeed_initialized()):
            with torch.cuda.amp.autocast(enabled=False):
                out = nn.functional.layer_norm(x, self.c_in, self.weight.to(dtype=d), self.bias.to(dtype=d), self.eps)
        else:
            out = nn.functional.layer_norm(x, self.c_in, self.weight, self.bias, self.eps)
        return out
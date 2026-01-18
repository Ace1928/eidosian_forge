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
def _prep_qkv(self, q_x: torch.Tensor, kv_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = self.linear_q(q_x)
    k = self.linear_k(kv_x)
    v = self.linear_v(kv_x)
    q = q.view(q.shape[:-1] + (self.no_heads, -1))
    k = k.view(k.shape[:-1] + (self.no_heads, -1))
    v = v.view(v.shape[:-1] + (self.no_heads, -1))
    q = q.transpose(-2, -3)
    k = k.transpose(-2, -3)
    v = v.transpose(-2, -3)
    q /= math.sqrt(self.c_hidden)
    return (q, k, v)
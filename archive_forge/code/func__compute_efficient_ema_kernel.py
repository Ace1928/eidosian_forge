import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_mega import MegaConfig
def _compute_efficient_ema_kernel(self, length: int):
    self._kernel = None
    damping_factor, previous_timestep_weight = self._compute_ema_coefficients()
    vander = torch.arange(length).to(damping_factor).view(1, 1, length) * torch.log(previous_timestep_weight)
    kernel = damping_factor * self.ema_expansion_matrix * torch.exp(vander)
    return torch.einsum('dnl,dn->dl', kernel, self.kernel_projection_matrix * self.scale)
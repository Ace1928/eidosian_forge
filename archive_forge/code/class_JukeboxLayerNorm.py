import math
import os
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm as FusedLayerNorm
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, logging
from ...utils.logging import tqdm
from .configuration_jukebox import ATTENTION_PATTERNS, JukeboxConfig, JukeboxPriorConfig, JukeboxVQVAEConfig
class JukeboxLayerNorm(FusedLayerNorm):

    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True):
        super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        self.width = np.prod(normalized_shape)
        self.max_numel = 65535 * self.width

    def forward(self, input):
        if input.numel() > self.max_numel:
            return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps).type_as(input)
        else:
            return super().forward(input).type_as(input)
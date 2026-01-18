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
class EsmFoldRelativePosition(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.bins = config.position_bins
        self.embedding = torch.nn.Embedding(2 * self.bins + 2, config.pairwise_state_dim)

    def forward(self, residue_index, mask=None):
        """
        Input:
          residue_index: B x L tensor of indices (dytpe=torch.long) mask: B x L tensor of booleans

        Output:
          pairwise_state: B x L x L x pairwise_state_dim tensor of embeddings
        """
        if residue_index.dtype != torch.long:
            raise ValueError(f'`residue_index` has dtype {residue_index.dtype}, it should be `torch.long`.')
        if mask is not None and residue_index.shape != mask.shape:
            raise ValueError(f'`residue_index` and `mask` have inconsistent shapes: {residue_index.shape} != {mask.shape}.')
        diff = residue_index[:, None, :] - residue_index[:, :, None]
        diff = diff.clamp(-self.bins, self.bins)
        diff = diff + self.bins + 1
        if mask is not None:
            mask = mask[:, None, :] * mask[:, :, None]
            diff[mask == False] = 0
        output = self.embedding(diff)
        return output
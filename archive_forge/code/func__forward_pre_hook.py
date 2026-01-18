import math
from typing import Iterable, Iterator, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_graphormer import GraphormerConfig
def _forward_pre_hook(mod, input):
    if mod.training:
        if not is_conv:
            weight = mod.weight
            in_features = weight.size(1)
            out_features = weight.size(0)
            mask = torch.zeros(in_features // block_size * out_features, device=weight.device)
            mask.bernoulli_(p)
            mask = mask.repeat_interleave(block_size, -1).view(-1, in_features)
        else:
            weight = mod.weight
            in_channels = mod.in_channels
            out_channels = mod.out_channels
            if mod.kernel_size == (1, 1):
                mask = torch.zeros(int(in_channels // block_size * out_channels), device=weight.device)
                mask.bernoulli_(p)
                mask = mask.repeat_interleave(block_size, -1).view(-1, in_channels)
            else:
                mask = torch.zeros(weight.size(0), weight.size(1), device=weight.device)
                mask.bernoulli_(p)
                mask = mask.unsqueeze(2).unsqueeze(3).repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])
        mask = mask.to(torch.bool)
        s = 1 / (1 - p)
        mod.weight.data = s * weight.masked_fill(mask, 0)
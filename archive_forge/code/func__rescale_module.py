import math
import typing as tp
from typing import Any, Dict, List, Optional
import torch
from torch import nn
from torch.nn import functional as F
def _rescale_module(module):
    """
    Rescales initial weight scale for all models within the module.
    """
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d)):
            std = sub.weight.std().detach()
            scale = (std / 0.1) ** 0.5
            sub.weight.data /= scale
            if sub.bias is not None:
                sub.bias.data /= scale
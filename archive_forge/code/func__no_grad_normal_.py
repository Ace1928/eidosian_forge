import math
import warnings
from torch import Tensor
import torch
from typing import Optional as _Optional
def _no_grad_normal_(tensor, mean, std, generator=None):
    with torch.no_grad():
        return tensor.normal_(mean, std, generator=generator)
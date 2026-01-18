from enum import Enum
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from xformers import _is_triton_available
from collections import namedtuple
class Skip(nn.Module):

    def __init__(self, *_, **__) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, **_):
        return x
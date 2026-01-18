import torch
from torch.nn import Module
from torch.ao.quantization.observer import (
import re
from abc import ABC, abstractmethod
from typing import Any, Tuple
def _is_symmetric_quant(qscheme: 'torch.qscheme') -> bool:
    return qscheme in [torch.per_tensor_symmetric, torch.per_channel_symmetric]
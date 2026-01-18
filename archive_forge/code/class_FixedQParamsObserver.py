import re
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from functools import partial
from typing import Any, List, Tuple, Optional, Dict
import torch
import torch.nn as nn
from torch.ao.quantization.utils import (
class FixedQParamsObserver(ObserverBase):
    """
    Observer that simulates quantize and dequantize with fixed
    quantization parameters in training time. Only per tensor
    quantization is supported.

    Args:
        `scale` (float): fixed scale for the observer
        `zero_point` (int): fixed zero point for the observer
        `dtype`, `qscheme`, `quant_min`, `quant_max`
    """
    scale: torch.Tensor
    zero_point: torch.Tensor

    def __init__(self, scale, zero_point, dtype=torch.quint8, qscheme=torch.per_tensor_affine, quant_min=0, quant_max=255, is_dynamic=False, **kwargs):
        if is_dynamic:
            raise NotImplementedError("FixedQParamsObserver doesn't support dynamic quantization")
        super().__init__(dtype=dtype, is_dynamic=is_dynamic, **kwargs)
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.register_buffer('scale', torch.tensor([scale], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([zero_point], dtype=torch.int))
        self.dtype = dtype
        self.qscheme = qscheme

    def forward(self, X):
        return X

    @torch.jit.export
    def calculate_qparams(self):
        return (self.scale, self.zero_point)
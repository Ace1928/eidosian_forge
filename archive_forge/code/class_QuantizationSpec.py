from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.ao.quantization import ObserverOrFakeQuantize
from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
from torch.fx import Node
@dataclass(eq=True, frozen=True)
class QuantizationSpec(QuantizationSpecBase):
    """Quantization spec for common operators that allows user to specify how to
    quantize a Tensor, this includes dtype, quant_min, quant_max etc.
    """
    dtype: torch.dtype
    observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor
    quant_min: Optional[int] = None
    quant_max: Optional[int] = None
    qscheme: Optional[torch.qscheme] = None
    ch_axis: Optional[int] = None
    is_dynamic: bool = False

    def __post_init__(self):
        if self.dtype not in SUPPORTED_DTYPES:
            raise TypeError(f'Unsupported dtype {self.dtype}.')
        if self.quant_min is not None and self.quant_max is not None and (self.quant_min > self.quant_max):
            raise ValueError(f'quant_min {self.quant_min} must be <= quant_max {self.quant_max}.')
        if self.qscheme is not None and self.qscheme not in SUPPORTED_QSCHEMES:
            raise ValueError(f'Unsupported qscheme {self.qscheme}.')
        if self.ch_axis is not None and self.ch_axis < 0:
            raise ValueError('Ch_axis is < 0.')
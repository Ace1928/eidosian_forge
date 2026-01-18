import re
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from functools import partial
from typing import Any, List, Tuple, Optional, Dict
import torch
import torch.nn as nn
from torch.ao.quantization.utils import (
class RecordingObserver(ObserverBase):
    """
    The module is mainly for debug and records the tensor values during runtime.

    Args:
        dtype: Quantized data type
        qscheme: Quantization scheme to be used
        reduce_range: Reduces the range of the quantized data type by 1 bit
    """
    __annotations__ = {'tensor_val': List[Optional[torch.Tensor]]}

    def __init__(self, dtype=torch.quint8):
        super().__init__(dtype=dtype, is_dynamic=False)
        self.tensor_val = []

    def forward(self, x):
        self.tensor_val.append(x.clone())
        return x

    @torch.jit.export
    def calculate_qparams(self):
        raise Exception('calculate_qparams should not be called for RecordingObserver')

    @torch.jit.export
    def get_tensor_value(self):
        return self.tensor_val
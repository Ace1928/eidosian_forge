import re
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from functools import partial
from typing import Any, List, Tuple, Optional, Dict
import torch
import torch.nn as nn
from torch.ao.quantization.utils import (
class ObserverBase(ABC, nn.Module):
    """Base observer Module.
    Any observer implementation should derive from this class.

    Concrete observers should follow the same API. In forward, they will update
    the statistics of the observed Tensor. And they should provide a
    `calculate_qparams` function that computes the quantization parameters given
    the collected statistics.

    Args:
        dtype: dtype argument to the `quantize` node needed to implement the
               reference model spec.
        is_dynamic: indicator for whether the observer is a placeholder for dynamic quantization
        or static quantization
    """

    def __init__(self, dtype, is_dynamic=False):
        super().__init__()
        self.dtype = dtype
        self.is_dynamic = is_dynamic

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def calculate_qparams(self, **kwargs):
        pass
    with_args = classmethod(_with_args)
    with_callable_args = classmethod(_with_callable_args)
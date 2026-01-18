from dataclasses import dataclass
from functools import reduce  # Required in Python 3
import operator
from typing import Callable, Optional, Tuple
import warnings
from warnings import warn
import torch
import bitsandbytes.functional as F
@dataclass
class MatmulLtState:
    _tile_indices: Optional[torch.Tensor] = None
    force_no_igemmlt: bool = False
    CB = None
    CxB = None
    SB = None
    SCB = None
    CxBt = None
    SBt = None
    CBt = None
    subB = None
    outlier_pool = None
    has_accumulated_gradients = False
    threshold = 0.0
    idx = None
    is_training = True
    has_fp16_weights = True
    memory_efficient_backward = False
    use_pool = False
    formatB = F.get_special_format_str()

    def reset_grads(self):
        self.CB = None
        self.CxB = None
        self.SB = None
        self.SCB = None
        self.CxBt = None
        self.SBt = None
        self.CBt = None

    @property
    def tile_indices(self):
        if self._tile_indices is None:
            self._tile_indices = get_tile_inds(self.formatB, self.CxB.device)
        return self._tile_indices
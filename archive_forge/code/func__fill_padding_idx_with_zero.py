import copy
from typing import Any, Dict, Optional, TypeVar, Union, overload
import warnings
import torch
from torch import Tensor, device, dtype, nn
import torch.nn.functional as F
import bitsandbytes as bnb
from bitsandbytes.autograd._functions import get_tile_inds, undo_layout
from bitsandbytes.functional import QuantState
from bitsandbytes.optim import GlobalOptimManager
from bitsandbytes.utils import OutlierTracer
def _fill_padding_idx_with_zero(self) -> None:
    if self.padding_idx is not None:
        with torch.no_grad():
            self.weight[self.padding_idx].fill_(0)
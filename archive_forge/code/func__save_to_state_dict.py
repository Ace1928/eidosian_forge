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
def _save_to_state_dict(self, destination, prefix, keep_vars):
    super()._save_to_state_dict(destination, prefix, keep_vars)
    scb_name = 'SCB'
    param_from_weight = getattr(self.weight, scb_name)
    param_from_state = getattr(self.state, scb_name)
    layout_reordered = self.state.CxB is not None
    key_name = prefix + f'{scb_name}'
    format_name = prefix + 'weight_format'
    if not self.state.has_fp16_weights:
        if param_from_weight is not None:
            destination[key_name] = param_from_weight if keep_vars else param_from_weight.detach()
            destination[format_name] = 'row'
        elif param_from_state is not None and (not layout_reordered):
            destination[key_name] = param_from_state if keep_vars else param_from_state.detach()
            destination[format_name] = 'row'
        elif param_from_state is not None:
            destination[key_name] = param_from_state if keep_vars else param_from_state.detach()
            destination[format_name] = self.state.formatB
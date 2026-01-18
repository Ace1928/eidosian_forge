from __future__ import annotations
import inspect
import warnings
from collections import abc, defaultdict
from enum import Enum
from typing import Any, cast, Dict, Iterable, List, Optional, overload, Tuple, Union
import torch
from .common import amp_definitely_not_available
def _check_inf_per_device(self, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    _scale, _ = self._check_scale_growth_tracker('_check_inf_per_device')
    dummy_inv_scale = torch.full((), 1.0, dtype=torch.float32, device=_scale.device)
    found_inf = torch.full((), 0.0, dtype=torch.float32, device=_scale.device)
    self._per_optimizer_states[id(optimizer)]['found_inf_per_device'] = self._unscale_grads_(optimizer, dummy_inv_scale, found_inf, True)
    return self._per_optimizer_states[id(optimizer)]['found_inf_per_device']
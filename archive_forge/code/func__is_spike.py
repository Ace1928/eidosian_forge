import json
import operator
import os
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union
import torch
from lightning_utilities.core.imports import compare_version
from lightning_fabric.utilities.types import _PATH
def _is_spike(self, loss: torch.Tensor) -> bool:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        running_val = self.running_mean.compute()
    curr_diff = loss - self.last_val
    if self.finite_only and (not torch.isfinite(loss)):
        return True
    if self._is_better(curr_diff):
        return False
    return self._check_atol(loss, running_val) and self._check_rtol(loss, running_val)
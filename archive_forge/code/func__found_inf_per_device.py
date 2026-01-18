from __future__ import annotations
import inspect
import warnings
from collections import abc, defaultdict
from enum import Enum
from typing import Any, cast, Dict, Iterable, List, Optional, overload, Tuple, Union
import torch
from .common import amp_definitely_not_available
def _found_inf_per_device(self, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    return self._per_optimizer_states[id(optimizer)]['found_inf_per_device']
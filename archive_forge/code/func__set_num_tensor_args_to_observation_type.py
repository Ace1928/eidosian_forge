from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, Union
import torch
from torch.ao.quantization.utils import Pattern
from enum import Enum
def _set_num_tensor_args_to_observation_type(self, num_tensor_args_to_observation_type: Dict[int, ObservationType]) -> BackendPatternConfig:
    self._num_tensor_args_to_observation_type = num_tensor_args_to_observation_type
    return self
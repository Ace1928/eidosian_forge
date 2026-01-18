from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, Union
import torch
from torch.ao.quantization.utils import Pattern
from enum import Enum
def _set_input_type_to_index(self, input_type_to_index: Dict[str, int]) -> BackendPatternConfig:
    self._input_type_to_index = input_type_to_index
    return self
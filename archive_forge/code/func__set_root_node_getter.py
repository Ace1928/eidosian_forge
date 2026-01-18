from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, Union
import torch
from torch.ao.quantization.utils import Pattern
from enum import Enum
def _set_root_node_getter(self, root_node_getter: Callable) -> BackendPatternConfig:
    self._root_node_getter = root_node_getter
    return self
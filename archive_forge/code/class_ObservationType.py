from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, Union
import torch
from torch.ao.quantization.utils import Pattern
from enum import Enum
class ObservationType(Enum):
    """ An enum that represents different ways of how an operator/operator pattern
    should be observed
    """
    OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT = 0
    'this means input and output are observed with different observers, based\n    on qconfig.activation\n    example: conv, linear, softmax\n    '
    OUTPUT_SHARE_OBSERVER_WITH_INPUT = 1
    'this means the output will use the same observer instance as input, based\n    on qconfig.activation\n    example: torch.cat, maxpool\n    '
    INPUT_OUTPUT_NOT_OBSERVED = 2
    'this means the input and output are never observed\n    example: x.shape, x.size\n    '
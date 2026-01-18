from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from functools import lru_cache
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Sequence, Set, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle
from fairscale.nn import FullyShardedDataParallel
class TraceBackwardEvent(NamedTuple):
    """
    Complementary trace event collected during the forward pass
    to trace the memory taken by activations
    """
    memory_activations: int

    def to_dict(self) -> Dict[str, Any]:
        return {'memory_activations': self.memory_activations}

    @classmethod
    def from_dict(cls, serialized: Dict[str, Any]) -> 'TraceBackwardEvent':
        return TraceBackwardEvent(memory_activations=serialized['memory_activations'])
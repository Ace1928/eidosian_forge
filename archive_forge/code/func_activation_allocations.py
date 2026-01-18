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
def activation_allocations(self, ax: Any, job_name: str, memory_traces: List[LayerMemoryTrace]) -> None:
    event_allocations = [t.event.memory_activations for t in memory_traces]
    x, y_forward, y_backward = self._split_forward_backward(memory_traces, event_allocations)
    ax.plot(x, y_forward, x, y_backward, label=job_name)
    self._y_axis_in_gigabytes(ax)
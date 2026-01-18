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
@staticmethod
def _capture_memory() -> Tuple[int, int]:
    torch.cuda.synchronize()
    allocated_mb = torch.cuda.memory_allocated()
    reserved_mb = torch.cuda.memory_reserved()
    return (allocated_mb, reserved_mb)
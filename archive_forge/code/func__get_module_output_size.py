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
@classmethod
def _get_module_output_size(cls, xs: Union[torch.Tensor, Sequence[torch.Tensor]]) -> int:
    """
        Return the minimum memory requirement to store the tensors
        provided as parameters
        """
    if isinstance(xs, torch.Tensor):
        x = xs
        p = cls._get_dtype_size(x)
        for d in x.shape:
            p *= d
        return p
    elif isinstance(xs, tuple) or isinstance(xs, list):
        return sum((cls._get_module_output_size(x) for x in xs))
    return 0
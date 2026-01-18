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
def _filter_allocated_output(cls, inputs: Union[torch.Tensor, Sequence[torch.Tensor]], outputs: Union[torch.Tensor, Sequence[torch.Tensor]]) -> List[torch.Tensor]:
    """
        Only return the outputs that are allocated and not views, reshape
        or stride of the inputs
        """
    xs = cls._collect_tensors(inputs)
    ys = cls._collect_tensors(outputs)
    return [y for y in ys if all((not cls._is_same_storage(x, y) for x in xs))]
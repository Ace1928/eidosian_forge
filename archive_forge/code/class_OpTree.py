import collections
import dataclasses
import enum
import itertools as it
import logging
from typing import (
from typing_extensions import Literal
import torch
from torch._C import FunctionSchema
from torch._C._autograd import _ProfilerResult
from torch._C._profiler import (
from torch._utils import _element_size
from torch.profiler import _utils
class OpTree:

    def __init__(self, result: _ProfilerResult) -> None:
        self._root_nodes = result.experimental_event_tree()
        self._sorted_nodes = tuple(sorted(self.dfs(), key=lambda x: x.start_time_ns))

    def dfs(self, *args, **kwargs) -> Iterator[_ProfilerEvent]:
        yield from _utils.traverse_dfs(self._root_nodes, *args, **kwargs)

    @property
    def sorted_nodes(self) -> Tuple[_ProfilerEvent, ...]:
        return self._sorted_nodes
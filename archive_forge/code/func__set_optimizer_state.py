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
def _set_optimizer_state(self) -> None:
    for event in self._op_tree.dfs():
        if event.typed[0] == _EventType.PyCall and event.typed[1].optimizer:
            parameters = event.typed[1].optimizer.parameters
            for _, t in it.chain(*[state for _, _, state in parameters]):
                key = TensorKey.from_tensor(t)
                if key is not None:
                    self._categories.set_by_id(key, Category.OPTIMIZER_STATE)
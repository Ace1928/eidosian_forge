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
def _set_autograd_detail(self):
    prior = {None, Category.AUTOGRAD_DETAIL}
    for node in self._data_flow_graph.flow_nodes:
        if RecordScope.BACKWARD_FUNCTION in get_scopes(node._event):
            for key, version in node.outputs.items():
                if version == 0 or self._categories.get(key, version - 1) in prior:
                    self._categories.setdefault_by_version(key, version, Category.AUTOGRAD_DETAIL)
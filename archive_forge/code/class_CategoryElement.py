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
@dataclasses.dataclass
class CategoryElement:
    by_id: Optional[Category] = None
    by_key: Dict[TensorKey, Category] = dataclasses.field(default_factory=dict)
    by_version: Dict[TensorAndID, Category] = dataclasses.field(default_factory=dict)
    _by_id_keyset: Set[TensorKey] = dataclasses.field(default_factory=set)
import collections
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, NewType, Optional, Set, Union
import torch
import torch.utils._pytree as pytree
from torch._guards import Source
from torch._subclasses import FakeTensor
from torch._subclasses.fake_tensor import is_fake
from .. import config
from .utils import strict_zip
class MutationType(Enum):
    NOT_MUTATED = 1
    MUTATED_IN_GRAPH = 2
    MUTATED_OUT_GRAPH = 3
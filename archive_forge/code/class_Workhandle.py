import collections.abc as abc
from dataclasses import dataclass
from math import inf
from typing import Any, Callable, Dict, List, Optional
import torch
import torch.distributed as dist
@dataclass
class Workhandle:
    handle: Any
    callback: Optional[Callable] = None
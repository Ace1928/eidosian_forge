import contextlib
from typing import Optional, Union, List, Set, Dict, Any
import warnings
from dataclasses import dataclass
import torch
import torchgen
from torch._C import _len_torch_dispatch_stack, _get_dispatch_stack_at,\
@dataclass
class SchemaInfo:
    args: List[AliasInfo]
    outs: List[AliasInfo]
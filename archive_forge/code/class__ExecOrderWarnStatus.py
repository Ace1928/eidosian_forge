import itertools
import warnings
from enum import auto, Enum
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed.fsdp._common_utils import _FSDPState, _get_param_to_fqns
from torch.distributed.fsdp._flat_param import FlatParamHandle
class _ExecOrderWarnStatus(Enum):
    """Used internally for execution order validation."""
    NONE = auto()
    WARNING = auto()
    WARNED = auto()
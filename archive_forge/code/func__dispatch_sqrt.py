import math
import functools
import warnings
from collections import OrderedDict, defaultdict
from copy import deepcopy
from itertools import chain
from typing import (
from typing_extensions import ParamSpec, Self, TypeAlias
import torch
import torch.utils.hooks as hooks
from torch.utils.hooks import RemovableHandle
from torch.utils._foreach_utils import (
from torch._utils import is_compiling
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype
def _dispatch_sqrt(x: float):
    if not torch.jit.is_scripting() and isinstance(x, torch.Tensor):
        return x.sqrt()
    else:
        return math.sqrt(x)
import math
from functools import wraps
from typing import Callable, Optional, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
import torch._refs as refs
from torch._decomp import register_decomposition
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._refs import _make_inplace
def _check_reduction_value(reduction: str):
    if reduction not in ('mean', 'sum', 'none'):
        raise ValueError(f'{reduction} is not a valid value for reduction')
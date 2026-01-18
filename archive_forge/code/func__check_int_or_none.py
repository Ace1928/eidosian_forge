import torch
import functools
import threading
from torch import Tensor
from typing import Any, Callable, Optional, Tuple, Union, List
from torch.utils._pytree import (
from functools import partial
import os
import itertools
from torch._C._functorch import (
def _check_int_or_none(x, func, out_dims):
    if isinstance(x, int):
        return
    if x is None:
        return
    raise ValueError(f'vmap({_get_name(func)}, ..., out_dims={out_dims}): `out_dims` must be an int, None or a python collection of ints representing where in the outputs the vmapped dimension should appear.')
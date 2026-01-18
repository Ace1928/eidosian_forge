import inspect
import warnings
from functools import wraps
from itertools import chain
from typing import Callable, NamedTuple, Optional, overload, Sequence, Tuple
import torch
import torch._prims_common as utils
from torch._prims_common import (
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_unflatten
def _maybe_resize_out(out: TensorLikeType, shape: ShapeType):
    if _resize_output_check(out, shape):
        return out.resize_(shape)
    else:
        return out
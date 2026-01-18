from typing import Callable, List, Optional, Tuple, Union
import math
import warnings
import importlib
import torch
from torch import _VF
from torch import sym_int as _sym_int
from torch._C import _infer_size, _add_docstr
from torch._torch_docs import reproducibility_notes, tf32_notes, sparse_support_notes
from typing import TYPE_CHECKING
from .._jit_internal import boolean_dispatch, _overload, BroadcastingList1, BroadcastingList2, BroadcastingList3
from ..overrides import (
from . import _reduction as _Reduction
from . import grad  # noqa: F401
from .modules import utils
from .modules.utils import _single, _pair, _triple, _list_with_default
def _is_integer(x) -> bool:
    """Type check the input number is an integer.

    Will return True for int, SymInt, Numpy integers and Tensors with integer elements.
    """
    if isinstance(x, (int, torch.SymInt)):
        return True
    if np is not None and isinstance(x, np.integer):
        return True
    return isinstance(x, Tensor) and (not x.is_floating_point())
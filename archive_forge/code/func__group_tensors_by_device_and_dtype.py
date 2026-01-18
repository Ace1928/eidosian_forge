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
@staticmethod
def _group_tensors_by_device_and_dtype(tensorlistlist: TensorListList, with_indices: bool=False) -> Union[Dict[Tuple[None, None], Tuple[TensorListList, Indices]], Dict[Tuple[torch.device, torch.dtype], Tuple[TensorListList, Indices]]]:
    """Groups a list of lists of tensors by device and dtype.
        Skips this step if we are compiling since this will occur during inductor lowering."""
    if is_compiling():
        return {(None, None): (tensorlistlist, list(range(len(tensorlistlist[0]))))}
    else:
        return _group_tensors_by_device_and_dtype(tensorlistlist, with_indices)
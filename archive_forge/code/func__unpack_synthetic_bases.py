import collections
import pprint
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.utils.dlpack
from torch import Tensor
from torch._guards import DuplicateInputs, TracingContext
from torch._prims_common import CUDARngStateHelper
from torch.multiprocessing.reductions import StorageWeakRef
from .. import config
from .collect_metadata_analysis import run_functionalized_fw_and_collect_metadata
from .functional_utils import gen_alias_from_base
from .input_output_analysis import (
from .logging_utils import describe_input, format_guard_bug_msg
from .schemas import (
from .subclass_utils import (
from .utils import (
def _unpack_synthetic_bases(primals: Tuple[Any, ...]) -> List[Any]:
    f_args_inner = []
    for inner_idx_or_tuple in synthetic_base_info:
        if isinstance(inner_idx_or_tuple, int):
            f_args_inner.append(primals[inner_idx_or_tuple])
        else:
            inner_base_idx, view_tensor = inner_idx_or_tuple
            base = primals[inner_base_idx]
            view_arg = gen_alias_from_base(base, view_tensor, view_tensor.requires_grad)
            f_args_inner.append(view_arg)
    return f_args_inner
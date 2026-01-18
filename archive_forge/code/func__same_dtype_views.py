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
def _same_dtype_views(view1, view2):
    if view1.dtype != view2.dtype:
        return False
    if view1._base is not None and view1.dtype != view1._base.dtype:
        return False
    if view2._base is not None and view2.dtype != view2._base.dtype:
        return False
    return True
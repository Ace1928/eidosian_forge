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
def _are_differentiable_views(view1, view2):
    if view1 is view2:
        return True
    if view1._base is None and view2._base is None:
        return False
    if view1._base is view2._base or view1._base is view2 or view1 is view2._base:
        return True
    return False
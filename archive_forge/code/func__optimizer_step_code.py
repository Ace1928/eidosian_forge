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
def _optimizer_step_code(self) -> None:
    """Entry point for `torch.profile.profiler`.

        When python tracing is enabled the profiler will hook into this
        function at the CPython level to inspect the optimizer's parameters and
        param groups. It is called it after `step()` since many optimizers
        lazily initialize state.

        This is a workaround due to lack of a proper step hook on the optimizer,
        and will be removed if it exists.
        """
    pass
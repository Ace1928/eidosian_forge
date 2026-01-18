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
def _patch_step_function(self) -> None:
    self._zero_grad_profile_name = f'Optimizer.zero_grad#{self.__class__.__name__}.zero_grad'
    hooked = getattr(self.__class__.step, 'hooked', None)
    if not hooked:
        self.__class__.step = self.profile_hook_step(self.__class__.step)
        self.__class__.step.hooked = True
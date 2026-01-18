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
def _cuda_graph_capture_health_check(self) -> None:
    if not is_compiling() and torch.backends.cuda.is_built() and torch.cuda.is_available():
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and (not all((group['capturable'] for group in self.param_groups))):
            raise RuntimeError('Attempting CUDA graph capture of step() for an instance of ' + self.__class__.__name__ + " but param_groups' capturable is False.")
        if not getattr(self, '_warned_capturable_if_run_uncaptured', False) and all((group['capturable'] for group in self.param_groups)) and (not capturing):
            warnings.warn('This instance was constructed with capturable=True or some of all the param_groups came with capturable=True, but step() is running without CUDA graph capture. If you never intend to graph-capture this instance, capturable=True can impair performance, and you should set capturable=False.')
            self._warned_capturable_if_run_uncaptured = True
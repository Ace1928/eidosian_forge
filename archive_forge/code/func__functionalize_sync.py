import copyreg
import functools
import sys
import traceback
import warnings
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, DefaultDict, List, Optional
import torch
def _functionalize_sync(t):
    from torch._subclasses.functional_tensor import FunctionalTensor, maybe_disable_functional_mode
    ctx = maybe_disable_functional_mode if isinstance(t, FunctionalTensor) else nullcontext
    if isinstance(t, FunctionalTensor):
        maybe_functional_mode = torch._C._unset_dispatch_mode(torch._C._TorchDispatchModeKey.FUNCTIONAL)
        try:
            torch._functionalize_sync(t.elem)
        finally:
            if maybe_functional_mode is not None:
                torch._C._set_dispatch_mode(maybe_functional_mode)
    else:
        torch._functionalize_sync(t)
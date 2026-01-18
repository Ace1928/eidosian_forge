import copyreg
import functools
import sys
import traceback
import warnings
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, DefaultDict, List, Optional
import torch
def _rebuild_tensor_v3(storage, storage_offset, size, stride, requires_grad, backward_hooks, dtype, metadata=None):
    t = torch.tensor([], dtype=dtype, device=storage._untyped_storage.device, requires_grad=requires_grad)
    t.set_(storage._untyped_storage, storage_offset, size, stride)
    if metadata:
        set_tensor_metadata(t, metadata)
    t._backward_hooks = backward_hooks
    return t
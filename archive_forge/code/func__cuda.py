import copyreg
import functools
import sys
import traceback
import warnings
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, DefaultDict, List, Optional
import torch
def _cuda(self, device=None, non_blocking=False, **kwargs):
    """Returns a copy of this object in CUDA memory.

    If this object is already in CUDA memory and on the correct device, then
    no copy is performed and the original object is returned.

    Args:
        device (int): The destination GPU id. Defaults to the current device.
        non_blocking (bool): If ``True`` and the source is in pinned memory,
            the copy will be asynchronous with respect to the host. Otherwise,
            the argument has no effect.
        **kwargs: For compatibility, may contain the key ``async`` in place of
            the ``non_blocking`` argument.
    """
    non_blocking = _get_async_or_non_blocking('cuda', non_blocking, kwargs)
    if self.is_cuda:
        if device is None:
            device = torch.cuda.current_device()
        if self.get_device() == device:
            return self
    elif device is None:
        device = -1
    with torch.cuda.device(device):
        if self.is_sparse:
            new_type = getattr(torch.cuda.sparse, self.__class__.__name__)
            indices = torch.Tensor._indices(self).cuda(device, non_blocking)
            values = torch.Tensor._values(self).cuda(device, non_blocking)
            return new_type(indices, values, self.size())
        else:
            untyped_storage = torch.UntypedStorage(self.size(), device=torch.device('cuda'))
            untyped_storage.copy_(self, non_blocking)
            return untyped_storage
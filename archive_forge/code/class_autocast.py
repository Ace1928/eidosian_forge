import collections
import functools
import torch
from typing import Any
class autocast(torch.amp.autocast_mode.autocast):
    """See :class:`torch.autocast`.

    ``torch.cuda.amp.autocast(args...)`` is equivalent to ``torch.autocast("cuda", args...)``
    """

    def __init__(self, enabled: bool=True, dtype: torch.dtype=torch.float16, cache_enabled: bool=True):
        if torch._jit_internal.is_scripting():
            self._enabled = enabled
            self.device = 'cuda'
            self.fast_dtype = dtype
            return
        super().__init__('cuda', enabled=enabled, dtype=dtype, cache_enabled=cache_enabled)

    def __enter__(self):
        if torch._jit_internal.is_scripting():
            return self
        return super().__enter__()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        if torch._jit_internal.is_scripting():
            return
        return super().__exit__(exc_type, exc_val, exc_tb)

    def __call__(self, func):
        if torch._jit_internal.is_scripting():
            return func
        return super().__call__(func)
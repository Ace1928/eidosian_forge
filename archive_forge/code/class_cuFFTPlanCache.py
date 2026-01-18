import contextlib
from typing import Union
import torch
from torch._C import _SDPAParams as SDPAParams, _SDPBackend as SDPBackend
class cuFFTPlanCache:
    """
    Represent a specific plan cache for a specific `device_index`.

    The attributes `size` and `max_size`, and method `clear`, can fetch and/ or
    change properties of the C++ cuFFT plan cache.
    """

    def __init__(self, device_index):
        self.device_index = device_index
    size = cuFFTPlanCacheAttrContextProp(torch._cufft_get_plan_cache_size, '.size is a read-only property showing the number of plans currently in the cache. To change the cache capacity, set cufft_plan_cache.max_size.')
    max_size = cuFFTPlanCacheAttrContextProp(torch._cufft_get_plan_cache_max_size, torch._cufft_set_plan_cache_max_size)

    def clear(self):
        return torch._cufft_clear_plan_cache(self.device_index)
import contextlib
from typing import Union
import torch
from torch._C import _SDPAParams as SDPAParams, _SDPBackend as SDPBackend
class cuFFTPlanCacheManager:
    """
    Represent all cuFFT plan caches, return the cuFFTPlanCache for a given device when indexed.

    Finally, this object, when used directly as a `cuFFTPlanCache` object (e.g.,
    setting the `.max_size`) attribute, the current device's cuFFT plan cache is
    used.
    """
    __initialized = False

    def __init__(self):
        self.caches = []
        self.__initialized = True

    def __getitem__(self, device):
        index = torch.cuda._utils._get_device_index(device)
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(f'cufft_plan_cache: expected 0 <= device index < {torch.cuda.device_count()}, but got device with index {index}')
        if len(self.caches) == 0:
            self.caches.extend((cuFFTPlanCache(index) for index in range(torch.cuda.device_count())))
        return self.caches[index]

    def __getattr__(self, name):
        return getattr(self[torch.cuda.current_device()], name)

    def __setattr__(self, name, value):
        if self.__initialized:
            return setattr(self[torch.cuda.current_device()], name, value)
        else:
            return super().__setattr__(name, value)
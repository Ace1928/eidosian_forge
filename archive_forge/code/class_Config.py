from __future__ import annotations
import builtins
import time
from typing import Dict
from ..testing import do_bench
from .jit import KernelInterface
class Config:
    """
    An object that represents a possible kernel configuration for the auto-tuner to try.

    :ivar meta: a dictionary of meta-parameters to pass to the kernel as keyword arguments.
    :type meta: dict[Str, Any]
    :ivar num_warps: the number of warps to use for the kernel when compiled for GPUs. For example, if
                      `num_warps=8`, then each kernel instance will be automatically parallelized to
                      cooperatively execute using `8 * 32 = 256` threads.
    :type num_warps: int
    :ivar num_stages: the number of stages that the compiler should use when software-pipelining loops.
                       Mostly useful for matrix multiplication workloads on SM80+ GPUs.
    :type enable_warp_specialization: bool
    :ivar enable_warp_specialization: enable specialization (spatial partitioning) or not. See https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#spatial-partitioning-also-known-as-warp-specialization
    :ivar pre_hook: a function that will be called before the kernel is called. Parameters of this
                    function are args.
    """

    def __init__(self, kwargs, num_warps=4, num_stages=2, num_ctas=1, enable_warp_specialization=False, pre_hook=None):
        self.kwargs = kwargs
        self.num_warps = num_warps
        self.num_ctas = num_ctas
        self.num_stages = num_stages
        self.enable_warp_specialization = enable_warp_specialization
        self.enable_persistent = False
        self.pre_hook = pre_hook

    def __str__(self):
        res = []
        for k, v in self.kwargs.items():
            res.append(f'{k}: {v}')
        res.append(f'num_warps: {self.num_warps}')
        res.append(f'num_ctas: {self.num_ctas}')
        res.append(f'num_stages: {self.num_stages}')
        res.append(f'enable_warp_specialization: {self.enable_warp_specialization}')
        res.append(f'enable_persistent: {self.enable_persistent}')
        return ', '.join(res)
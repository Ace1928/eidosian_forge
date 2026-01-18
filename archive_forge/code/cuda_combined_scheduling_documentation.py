from typing import List
from ..scheduler import BaseSchedulerNode, BaseScheduling, Scheduler, SchedulerNode
from .cuda.cuda_cpp_scheduling import CUDACPPScheduling
from .triton import TritonScheduling

    Scheduler for CUDA Kernels, which delegates calls as appropriate
    to the CUDA-C++ and Triton Schedulers, which both work for CUDA devices
    and use a unified-wrapper for codegen.

    If Scheduling code needs to be specialized for the case of mixed Triton / CUDA C++ code,
    this would also be the place to do it.
    
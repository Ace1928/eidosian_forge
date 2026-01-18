import numpy
import warnings
import cupy
from cupy.cuda import nccl
from cupyx.distributed import _store
from cupyx.distributed._comm import _Backend
from cupyx.scipy import sparse
Performs a barrier operation.

        The barrier is done in the cpu and is a explicit synchronization
        mechanism that halts the thread progression.
        
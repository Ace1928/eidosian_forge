import contextlib
import importlib
import os
import sys
import threading
import traceback
import warnings
from functools import lru_cache
from typing import Any, cast, List, Optional, Tuple, Union
import torch
import torch._C
from torch.types import Device
from .. import device as _device
from .._utils import classproperty
from ._utils import _dummy_type, _get_device_index
from .graphs import (
from .streams import Event, ExternalStream, Stream
from .memory import *  # noqa: F403
from .random import *  # noqa: F403
from torch.storage import _LegacyStorage, _warn_typed_storage_removal
from . import amp, jiterator, nvtx, profiler, sparse
def _register_triton_kernels():
    if torch._running_with_deploy():
        return

    @_WrappedTritonKernel
    def kernel_impl(*args, **kwargs):
        from torch.sparse._triton_ops import bsr_dense_mm
        return bsr_dense_mm(*args, skip_checks=True, **kwargs)

    @_WrappedTritonKernel
    def addmm_kernel_impl(*args, **kwargs):
        from torch.sparse._triton_ops import bsr_dense_addmm
        return bsr_dense_addmm(*args, skip_checks=True, **kwargs)
    has_triton = importlib.util.find_spec('triton') is not None
    if has_triton:
        torch._TritonLibrary.registerOp('_triton_bsr_dense_mm_out', '_triton_bsr_dense_mm_out(Tensor bsr, Tensor dense, *, Tensor(a!) out) -> Tensor(a!)', kernel_impl, 'SparseCsrCUDA')
        torch._TritonLibrary.registerOp('_triton_bsr_dense_addmm_out', '_triton_bsr_dense_addmm_out(Tensor input, Tensor bsr, Tensor dense, *, Scalar beta, Scalar alpha, Tensor(a!) out) -> Tensor(a!)', addmm_kernel_impl, 'SparseCsrCUDA')
from __future__ import annotations
import contextlib
import dataclasses
import functools
import logging
import os
import queue
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from ctypes import byref, c_size_t, c_void_p
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue
from typing import (
import torch
from torch import multiprocessing
from torch._dynamo.testing import rand_strided
from torch._inductor import ir
from torch._inductor.codecache import CUDACodeCache, DLLWrapper, PyCodeCache
from . import config
from .utils import do_bench
from .virtualized import V
@dataclasses.dataclass
class BenchmarkRequest:
    """
    Only handle triton template benchmark for now. The extern kernel benchmark
    can be done inside the same process since they usually don't cause crash.
    """

    def __init__(self, kernel_name: str, input_tensor_meta: Union[TensorMeta, List[TensorMeta]], output_tensor_meta: Union[TensorMeta, List[TensorMeta]], extra_args: Iterable[Any]):
        self.kernel_name = kernel_name
        if isinstance(input_tensor_meta, TensorMeta):
            input_tensor_meta = [input_tensor_meta]
        self.input_tensor_meta = input_tensor_meta
        if isinstance(output_tensor_meta, (tuple, list)):
            assert len(output_tensor_meta) == 1
            output_tensor_meta = output_tensor_meta[0]
        self.output_tensor_meta = output_tensor_meta
        self.extra_args = extra_args

    def make_run_fn(self, *input_tensors: torch.Tensor, output_tensor: torch.Tensor) -> Callable[[], None]:
        raise NotImplementedError()

    def cleanup_run_fn(self) -> None:
        pass

    def benchmark(self, *input_tensors: torch.Tensor, output_tensor: Optional[torch.Tensor]=None) -> float:
        debug = log.isEnabledFor(logging.DEBUG)
        if debug:
            start_ts = time.time()
        if output_tensor is None:
            assert len(input_tensors) == 0
            input_tensors = tuple((x.to_tensor() for x in self.input_tensor_meta))
            output_tensor = self.output_tensor_meta.to_tensor()
        if debug:
            create_tensor_elapse = time.time() - start_ts
            start_ts = time.time()
        fn = self.make_run_fn(*input_tensors, output_tensor=output_tensor)
        if debug:
            load_elapse = time.time() - start_ts
            start_ts = time.time()
        out = do_bench(fn)
        torch.cuda.synchronize()
        if debug:
            bench_elapse = time.time() - start_ts
            log.debug('InChildProcess %s: load %f, create tensor %f, bench %f', str(self), load_elapse, create_tensor_elapse, bench_elapse)
        self.cleanup_run_fn()
        return out
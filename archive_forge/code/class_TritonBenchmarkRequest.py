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
class TritonBenchmarkRequest(BenchmarkRequest):

    def __init__(self, kernel_name: str, input_tensor_meta: Union[TensorMeta, List[TensorMeta]], output_tensor_meta: Union[TensorMeta, List[TensorMeta]], extra_args: Iterable[Any], module_path: str, module_cache_key: str, grid: List[int], num_stages: int, num_warps: int):
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, extra_args)
        self.module_path = module_path
        self.module_cache_key = module_cache_key
        self.grid = grid
        self.num_stages = num_stages
        self.num_warps = num_warps

    def make_run_fn(self, *input_tensors: torch.Tensor, output_tensor: torch.Tensor) -> Callable[[], None]:
        mod = PyCodeCache.load_by_key_path(self.module_cache_key, self.module_path)
        log.debug('benchmark module key: %s, path: %s', self.module_cache_key, self.module_path)
        run_method = getattr(mod, self.kernel_name).run
        return functools.partial(run_method, *input_tensors, output_tensor, *self.extra_args, grid=self.grid, num_stages=self.num_stages, num_warps=self.num_warps, stream=torch.cuda.current_stream().cuda_stream)

    def __str__(self) -> str:
        return f'self.kernel_name={self.kernel_name!r}, self.module_path={self.module_path!r}, self.module_cache_key={self.module_cache_key!r}'
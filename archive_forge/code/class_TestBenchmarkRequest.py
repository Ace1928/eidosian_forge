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
class TestBenchmarkRequest(BenchmarkRequest):
    """
    Supports unit testing. Defined in this file so that the TuningProcess
    sub-process knows how to unpickle these objects.
    """

    def __init__(self, value: Optional[float]=None) -> None:
        self.value = value

    def benchmark(self, *input_tensors: torch.Tensor, output_tensor: Optional[torch.Tensor]=None) -> float:
        if self.value is None:
            raise Exception('Failed to run')
        return self.value
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
class TuningProcess:
    """
    Abstraction for launching a helper process to benchmark kernels. Spawns
    the parent process and uses multiprocessing queues to send benchmark
    requests and return results.
    """
    device: Optional[int] = None
    process: Optional[BaseProcess] = None
    request_queue: Optional[Queue[Any]] = None
    response_queue: Optional[Queue[Any]] = None

    @staticmethod
    def process_main(request_queue: Queue[Any], response_queue: Queue[Any]) -> None:
        """
        Entry point for the child process.
        """
        log.debug('Entering TuningProcess child. Visible devices = %s', os.environ.get(CUDA_VISIBLE_DEVICES))
        try:
            TuningProcess.workloop(request_queue, response_queue)
        except Exception as ex:
            log.exception('Exception in TuningProcess: %s', ex)

    @staticmethod
    def workloop(request_queue: Queue[Any], response_queue: Queue[Any]) -> None:
        """
        Work loop for the benchmarking subprocess.
        """
        while True:
            obj = request_queue.get()
            if obj is None:
                break
            elif isinstance(obj, Ping):
                response_queue.put(Pong())
            elif isinstance(obj, BenchmarkRequest):
                response_queue.put(obj.benchmark())
            else:
                raise RuntimeError(f'Invalid request type {type(obj)}')

    def valid(self) -> bool:
        """
        True if the sub-process has been initialized.
        """
        return self.process is not None and self.request_queue is not None and (self.response_queue is not None)

    def clear(self) -> None:
        """
        Reset to an uninitialized state.
        """
        self.process = self.request_queue = self.response_queue = None

    def initialize(self) -> None:
        """
        Create child process, request/response queues, and do the warm up.
        Set the environment to make only the provided GPU device visible
        to the process.
        """
        if self.valid():
            return
        ctx = multiprocessing.get_context('spawn')
        self.request_queue = ctx.Queue()
        self.response_queue = ctx.Queue()
        self.process = ctx.Process(target=self.process_main, args=(self.request_queue, self.response_queue))
        assert self.process is not None
        with set_cuda_visible_device(self.device):
            self.process.start()

    def put(self, obj: Any) -> None:
        """
        Push a work item to the child process.
        """
        self.initialize()
        assert self.request_queue is not None
        self.request_queue.put(obj)

    def get(self) -> Any:
        """
        Get a response from the child process.
        """
        assert self.process is not None
        assert self.response_queue is not None
        while True:
            try:
                return self.response_queue.get(timeout=1.0)
            except queue.Empty:
                status = self.process.exitcode
                if status is None:
                    continue
                self.clear()
                raise

    def terminate(self) -> None:
        """
        Signal the child process to terminate.
        """
        if self.valid():
            assert self.process is not None
            assert self.request_queue is not None
            self.request_queue.put(None)

    def wait(self) -> None:
        """
        Wait for the child process to exit.
        """
        if self.process is not None:
            self.process.join()
            self.clear()
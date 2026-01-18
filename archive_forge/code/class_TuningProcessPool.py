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
class TuningProcessPool:
    """
    Maintains a pool of TuningProcesses to benchmark kernels in parallel
    across devices. By default, we create one TuningProcess per device and
    set the sub-process environment to make only that device visible.
    """
    processes: Optional[queue.Queue[TuningProcess]] = None
    executor: Optional[ThreadPoolExecutor] = None

    def initialize(self) -> None:
        """
        Start the child processes.
        """
        assert (self.processes is None) == (self.executor is None)
        if self.processes is not None:
            return
        devices = self.get_device_list()
        log.debug('Sub-process autotune device list: %s', devices)
        self.processes = queue.Queue()
        for device in devices:
            p = TuningProcess(device=device)
            p.initialize()
            p.put(Ping())
            self.processes.put(p)
        for p in self.processes.queue:
            assert isinstance(p.get(), Pong)
        self.executor = ThreadPoolExecutor(max_workers=len(devices))
        global EXIT_HANDLER_REGISTERED
        if not EXIT_HANDLER_REGISTERED:
            EXIT_HANDLER_REGISTERED = True
            import atexit
            atexit.register(self.terminate)

    def get_device_list(self) -> Sequence[Optional[int]]:
        """
        Gather the list of devices to be used in the pool.
        """
        if not config.autotune_multi_device:
            return [None]
        count = torch.cuda.device_count()
        if CUDA_VISIBLE_DEVICES in os.environ:
            devices = [int(d) for d in os.environ[CUDA_VISIBLE_DEVICES].split(',')]
            assert len(devices) <= count
            return devices
        return list(range(count))

    def terminate(self) -> None:
        """
        Signal all child processes to terminate.
        """
        if self.executor is not None:
            self.executor.shutdown()
            self.executor = None
        if self.processes is not None:
            for p in self.processes.queue:
                p.terminate()
            for p in self.processes.queue:
                p.wait()
            self.processes = None

    def target(self, choice: TritonTemplateCaller) -> float:
        """
        Entry point for the thread-pool helper threads: Wait for an open TuningProcess,
        remove it from the queue, execute the benchmark in that subprocess, and return
        the TuningProcess to the queue.
        """
        assert choice.bmreq is not None
        assert self.processes is not None
        process = self.processes.get()
        process.put(choice.bmreq)
        try:
            return process.get()
        except queue.Empty:
            warnings.warn(f"Failed to benchmark choice '{choice}'. It will be ignored. Please debug the root cause in case the choice can bring perf gains.")
            return float('inf')
        finally:
            self.processes.put(process)

    def benchmark(self, choices: List[TritonTemplateCaller]) -> Dict[TritonTemplateCaller, float]:
        """
        Benchmark each choice in a separate process.
        """
        assert self.processes is not None, 'Tuning process pool is not initialized'
        assert self.executor is not None
        results = {}
        for choice, result in zip(choices, self.executor.map(self.target, choices)):
            results[choice] = result
        return results
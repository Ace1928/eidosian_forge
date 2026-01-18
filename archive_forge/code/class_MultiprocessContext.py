import abc
import logging
import os
import re
import signal
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import IntFlag
from multiprocessing import synchronize
from types import FrameType
from typing import Any, Callable, Dict, Optional, Set, Tuple, Union
import torch.multiprocessing as mp
from torch.distributed.elastic.multiprocessing.errors import ProcessFailure, record
from torch.distributed.elastic.multiprocessing.redirects import (
from torch.distributed.elastic.multiprocessing.tail_log import TailLog
class MultiprocessContext(PContext):
    """``PContext`` holding worker processes invoked as a function."""

    def __init__(self, name: str, entrypoint: Callable, args: Dict[int, Tuple], envs: Dict[int, Dict[str, str]], stdouts: Dict[int, str], stderrs: Dict[int, str], tee_stdouts: Dict[int, str], tee_stderrs: Dict[int, str], error_files: Dict[int, str], start_method: str, log_line_prefixes: Optional[Dict[int, str]]=None):
        super().__init__(name, entrypoint, args, envs, stdouts, stderrs, tee_stdouts, tee_stderrs, error_files, log_line_prefixes)
        self.start_method = start_method
        self._ret_vals = {local_rank: mp.get_context(self.start_method).SimpleQueue() for local_rank in range(self.nprocs)}
        self._return_values: Dict[int, Any] = {}
        self._pc: Optional[mp.ProcessContext] = None
        self._worker_finished_event = mp.get_context(self.start_method).Event()

    def _start(self):
        if self._pc:
            raise ValueError('The process context already initialized. Most likely the start method got called twice.')
        self._pc = mp.start_processes(fn=_wrap, args=(self.entrypoint, self.args, self.envs, self.stdouts, self.stderrs, self._ret_vals, self._worker_finished_event), nprocs=self.nprocs, join=False, daemon=False, start_method=self.start_method)

    def _is_done(self) -> bool:
        return len(self._return_values) == self.nprocs

    def _poll(self) -> Optional[RunProcsResult]:
        assert self._pc is not None
        try:
            self._pc.join(-1)
            for local_rank in range(0, self.nprocs):
                return_queue = self._ret_vals[local_rank]
                if not return_queue.empty():
                    self._return_values[local_rank] = return_queue.get()
            if self._is_done():
                self._worker_finished_event.set()
                self._pc.join()
                _validate_full_rank(self._return_values, self.nprocs, 'return_value queue')
                self.close()
                return RunProcsResult(return_values=self._return_values, stdouts=self.stdouts, stderrs=self.stderrs)
            else:
                return None
        except (mp.ProcessRaisedException, mp.ProcessExitedException) as e:
            failed_local_rank = e.error_index
            fn_name = self.entrypoint.__qualname__
            failed_proc = self._pc.processes[failed_local_rank]
            error_filepath = self.error_files[failed_local_rank]
            log.exception('failed (exitcode: %s) local_rank: %s (pid: %s) of fn: %s (start_method: %s)', failed_proc.exitcode, failed_local_rank, e.pid, fn_name, self.start_method)
            self.close()
            return RunProcsResult(failures={failed_local_rank: ProcessFailure(local_rank=failed_local_rank, pid=e.pid, exitcode=failed_proc.exitcode, error_file=error_filepath)}, stdouts=self.stdouts, stderrs=self.stderrs)

    def pids(self) -> Dict[int, int]:
        assert self._pc is not None
        return dict(enumerate(self._pc.pids()))

    def _close(self, death_sig: signal.Signals, timeout: int=30) -> None:
        if not self._pc:
            return
        for proc in self._pc.processes:
            if proc.is_alive():
                log.warning('Closing process %s via signal %s', proc.pid, death_sig.name)
                try:
                    os.kill(proc.pid, death_sig)
                except ProcessLookupError:
                    pass
        end = time.monotonic() + timeout
        for proc in self._pc.processes:
            time_to_wait = end - time.monotonic()
            if time_to_wait <= 0:
                break
            proc.join(time_to_wait)
        for proc in self._pc.processes:
            if proc.is_alive():
                log.warning('Unable to shutdown process %s via %s, forcefully exiting via %s', proc.pid, death_sig, _get_kill_signal())
                try:
                    os.kill(proc.pid, _get_kill_signal())
                except ProcessLookupError:
                    pass
            proc.join()
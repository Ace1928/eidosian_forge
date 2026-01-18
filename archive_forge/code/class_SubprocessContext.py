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
class SubprocessContext(PContext):
    """``PContext`` holding worker processes invoked as a binary."""

    def __init__(self, name: str, entrypoint: str, args: Dict[int, Tuple], envs: Dict[int, Dict[str, str]], stdouts: Dict[int, str], stderrs: Dict[int, str], tee_stdouts: Dict[int, str], tee_stderrs: Dict[int, str], error_files: Dict[int, str], log_line_prefixes: Optional[Dict[int, str]]=None):
        super().__init__(name, entrypoint, args, envs, stdouts, stderrs, tee_stdouts, tee_stderrs, error_files, log_line_prefixes)
        self._running_local_ranks: Set[int] = set(range(self.nprocs))
        self._failures: Dict[int, ProcessFailure] = {}
        self.subprocess_handlers: Dict[int, SubprocessHandler] = {}

    def _start(self):
        if self.subprocess_handlers:
            raise ValueError('The subprocess handlers already initialized. Most likely the start method got called twice.')
        self.subprocess_handlers = {local_rank: SubprocessHandler(entrypoint=self.entrypoint, args=self.args[local_rank], env=self.envs[local_rank], stdout=self.stdouts[local_rank], stderr=self.stderrs[local_rank]) for local_rank in range(self.nprocs)}

    def _poll(self) -> Optional[RunProcsResult]:
        done_local_ranks = set()
        for local_rank in self._running_local_ranks:
            handler = self.subprocess_handlers[local_rank]
            exitcode = handler.proc.poll()
            if exitcode is not None:
                done_local_ranks.add(local_rank)
                if exitcode != 0:
                    self._failures[local_rank] = ProcessFailure(local_rank=local_rank, pid=handler.proc.pid, exitcode=exitcode, error_file=self.error_files[local_rank])
        self._running_local_ranks.difference_update(done_local_ranks)
        if not self._running_local_ranks or self._failures:
            self.close()
            result = RunProcsResult(failures=self._failures, stdouts=self.stdouts, stderrs=self.stderrs)
            if result.is_failed():
                first_failure = min(result.failures.values(), key=lambda f: f.timestamp)
                log.error('failed (exitcode: %s) local_rank: %s (pid: %s) of binary: %s', first_failure.exitcode, first_failure.local_rank, first_failure.pid, self.entrypoint)
            else:
                result.return_values = {local_rank: None for local_rank in range(self.nprocs)}
            return result
        else:
            return None

    def pids(self) -> Dict[int, int]:
        return {local_rank: sh.proc.pid for local_rank, sh in self.subprocess_handlers.items()}

    def _close(self, death_sig: signal.Signals, timeout: int=30) -> None:
        if not self.subprocess_handlers:
            return
        for handler in self.subprocess_handlers.values():
            if handler.proc.poll() is None:
                log.warning('Sending process %s closing signal %s', handler.proc.pid, death_sig.name)
                handler.close(death_sig=death_sig)
        end = time.monotonic() + timeout
        for handler in self.subprocess_handlers.values():
            time_to_wait = end - time.monotonic()
            if time_to_wait <= 0:
                break
            try:
                handler.proc.wait(time_to_wait)
            except subprocess.TimeoutExpired:
                pass
        for handler in self.subprocess_handlers.values():
            if handler.proc.poll() is None:
                log.warning('Unable to shutdown process %s via %s, forcefully exiting via %s', handler.proc.pid, death_sig, _get_kill_signal())
                handler.close(death_sig=_get_kill_signal())
                handler.proc.wait()
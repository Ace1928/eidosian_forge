import json
import os
import signal
import socket
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from string import Template
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar
from torch.distributed.elastic.utils.logging import get_logger
from .error_handler import ErrorHandler  # noqa: F401
from .handlers import get_error_handler  # noqa: F401
class ChildFailedError(Exception):
    """
    Special exception type that can be raised from a function annotated with the
    ``@record`` decorator to have the child process' (root exception) propagate
    up the stack as-is (e.g. without being wrapped in the parent's traceback).

    Useful in cases where the parent is a simple nanny process
    and the child (worker) processes are actually doing meaningful compute.
    In this case, errors typically occur on the child process as the parent
    is not doing anything non-trivial, and child errors should be propagated
    to the scheduler for accurate root cause diagnostics.

    .. note:: The propagation relies on error files rather than exception handling to
              support both function and binary launches.

    Example:
    ::

     # process tree on a host (container)
     0: scheduler-init-process:
                |- 1: torchelastic_agent:
                         |- 2: trainer_0 (ok)
                         |- 3: trainer_1 (fail) -> error.json
                         |- ...
                         |- n+2: trainer_n (ok)
                |- n+3: other processes
                |- ...

    In the example above, trainer 1's failure (written into error.json) is
    the root cause and should be reported to the scheduler's init process.
    The torchelastic agent raises a ``ChildFailedError("trainer", {1: "trainer_1/error.json"})``
    upon detecting trainer 1's failure which would propagate the contents
    of trainer 1's error file to the scheduler's init process.
    """

    def __init__(self, name: str, failures: Dict[GlobalRank, ProcessFailure]):
        self.name = name
        self.failures = failures
        assert self.failures
        super().__init__(self.format_msg())

    def get_first_failure(self) -> Tuple[GlobalRank, ProcessFailure]:
        rank = min(self.failures.keys(), key=lambda r: self.failures[r].timestamp)
        return (rank, self.failures[rank])

    def format_msg(self, boarder_delim='=', section_delim='-'):
        title = f'{self.name} FAILED'
        root_rank, root_failure = self.get_first_failure()
        root_failure_fmt: str = ''
        other_failures_fmt: List[str] = []
        width = len(title)
        for idx, (rank, failure) in enumerate(self.failures.items()):
            fmt, w = self._format_failure(idx, rank, failure)
            width = max(width, w)
            if rank == root_rank:
                root_failure_fmt = fmt
            else:
                other_failures_fmt.append(fmt)
        width = min(width, 60)
        return Template(_MSG_FORMAT_TEMPLATE).substitute(boarder=boarder_delim * width, title=title, section=section_delim * width, root_failure=root_failure_fmt, other_failures='\n'.join(other_failures_fmt or ['  <NO_OTHER_FAILURES>']))

    def _format_failure(self, idx: int, rank: int, failure: ProcessFailure) -> Tuple[str, int]:
        msg = failure.message
        if isinstance(failure.message, dict):
            msg = failure.message.get('extraInfo', {}).get('py_callstack', failure.message.get('message', '<N/A>')).replace('\n', '\n  ')
        fmt = Template(_FAILURE_FORMAT_TEMPLATE).substitute(idx=idx, time=failure.timestamp_isoformat(), hostname=socket.getfqdn(), rank=rank, local_rank=failure.local_rank, exitcode=failure.exitcode, pid=failure.pid, error_file=failure.error_file, message=msg)
        width = 0
        for line in fmt.split('\n'):
            width = max(width, len(line))
        return (fmt, width)
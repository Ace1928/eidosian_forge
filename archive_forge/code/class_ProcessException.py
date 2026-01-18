import multiprocessing
import multiprocessing.connection
import signal
import sys
import warnings
from typing import Optional
from . import _prctl_pr_set_pdeathsig  # type: ignore[attr-defined]
class ProcessException(Exception):
    __slots__ = ['error_index', 'error_pid']

    def __init__(self, msg: str, error_index: int, pid: int):
        super().__init__(msg)
        self.msg = msg
        self.error_index = error_index
        self.pid = pid

    def __reduce__(self):
        return (type(self), (self.msg, self.error_index, self.pid))
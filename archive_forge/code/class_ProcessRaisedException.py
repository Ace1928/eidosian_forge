import multiprocessing
import multiprocessing.connection
import signal
import sys
import warnings
from typing import Optional
from . import _prctl_pr_set_pdeathsig  # type: ignore[attr-defined]
class ProcessRaisedException(ProcessException):
    """Exception raised when a process failed due to an exception raised by the code."""

    def __init__(self, msg: str, error_index: int, error_pid: int):
        super().__init__(msg, error_index, error_pid)
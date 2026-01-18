import os
import re
from .._core import SHELL_NAMES, ShellDetectionFailure
from . import proc, ps
def _iter_process_parents(pid, max_depth=10):
    """Select a way to obtain process information from the system.

    * `/proc` is used if supported.
    * The system `ps` utility is used as a fallback option.
    """
    for impl in (proc, ps):
        try:
            iterator = impl.iter_process_parents(pid, max_depth)
        except EnvironmentError:
            continue
        return iterator
    raise ShellDetectionFailure('compatible proc fs or ps utility is required')
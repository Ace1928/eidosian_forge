import re
import operator
import heapq
from collections import namedtuple
from collections.abc import Sequence
from contextlib import contextmanager
from functools import cached_property
from numba.core import config
import llvmlite.binding as llvm
class RecordLLVMPassTimings:
    """A helper context manager to track LLVM pass timings.
    """
    __slots__ = ['_data']

    def __enter__(self):
        """Enables the pass timing in LLVM.
        """
        llvm.set_time_passes(True)
        return self

    def __exit__(self, exc_val, exc_type, exc_tb):
        """Reset timings and save report internally.
        """
        self._data = llvm.report_and_reset_timings()
        llvm.set_time_passes(False)
        return

    def get(self):
        """Retrieve timing data for processing.

        Returns
        -------
        timings: ProcessedPassTimings
        """
        return ProcessedPassTimings(self._data)
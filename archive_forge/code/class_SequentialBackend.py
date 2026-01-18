import gc
import os
import warnings
import threading
import contextlib
from abc import ABCMeta, abstractmethod
from ._multiprocessing_helpers import mp
class SequentialBackend(ParallelBackendBase):
    """A ParallelBackend which will execute all batches sequentially.

    Does not use/create any threading objects, and hence has minimal
    overhead. Used when n_jobs == 1.
    """
    uses_threads = True
    supports_timeout = False
    supports_retrieve_callback = False
    supports_sharedmem = True

    def effective_n_jobs(self, n_jobs):
        """Determine the number of jobs which are going to run in parallel"""
        if n_jobs == 0:
            raise ValueError('n_jobs == 0 in Parallel has no meaning')
        return 1

    def apply_async(self, func, callback=None):
        """Schedule a func to be run"""
        raise RuntimeError('Should never be called for SequentialBackend.')

    def retrieve_result_callback(self, out):
        raise RuntimeError('Should never be called for SequentialBackend.')

    def get_nested_backend(self):
        from .parallel import get_active_backend
        return get_active_backend()
import gc
import os
import warnings
import threading
import contextlib
from abc import ABCMeta, abstractmethod
from ._multiprocessing_helpers import mp
class MultiprocessingBackend(PoolManagerMixin, AutoBatchingMixin, ParallelBackendBase):
    """A ParallelBackend which will use a multiprocessing.Pool.

    Will introduce some communication and memory overhead when exchanging
    input and output data with the with the worker Python processes.
    However, does not suffer from the Python Global Interpreter Lock.
    """
    supports_retrieve_callback = True
    supports_return_generator = False

    def effective_n_jobs(self, n_jobs):
        """Determine the number of jobs which are going to run in parallel.

        This also checks if we are attempting to create a nested parallel
        loop.
        """
        if mp is None:
            return 1
        if mp.current_process().daemon:
            if n_jobs != 1:
                if inside_dask_worker():
                    msg = "Inside a Dask worker with daemon=True, setting n_jobs=1.\nPossible work-arounds:\n- dask.config.set({'distributed.worker.daemon': False})- set the environment variable DASK_DISTRIBUTED__WORKER__DAEMON=False\nbefore creating your Dask cluster."
                else:
                    msg = 'Multiprocessing-backed parallel loops cannot be nested, setting n_jobs=1'
                warnings.warn(msg, stacklevel=3)
            return 1
        if process_executor._CURRENT_DEPTH > 0:
            if n_jobs != 1:
                warnings.warn('Multiprocessing-backed parallel loops cannot be nested, below loky, setting n_jobs=1', stacklevel=3)
            return 1
        elif not (self.in_main_thread() or self.nesting_level == 0):
            if n_jobs != 1:
                warnings.warn('Multiprocessing-backed parallel loops cannot be nested below threads, setting n_jobs=1', stacklevel=3)
            return 1
        return super(MultiprocessingBackend, self).effective_n_jobs(n_jobs)

    def configure(self, n_jobs=1, parallel=None, prefer=None, require=None, **memmappingpool_args):
        """Build a process or thread pool and return the number of workers"""
        n_jobs = self.effective_n_jobs(n_jobs)
        if n_jobs == 1:
            raise FallbackToBackend(SequentialBackend(nesting_level=self.nesting_level))
        gc.collect()
        self._pool = MemmappingPool(n_jobs, **memmappingpool_args)
        self.parallel = parallel
        return n_jobs

    def terminate(self):
        """Shutdown the process or thread pool"""
        super(MultiprocessingBackend, self).terminate()
        self.reset_batch_stats()
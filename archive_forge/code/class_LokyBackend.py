import gc
import os
import warnings
import threading
import contextlib
from abc import ABCMeta, abstractmethod
from ._multiprocessing_helpers import mp
class LokyBackend(AutoBatchingMixin, ParallelBackendBase):
    """Managing pool of workers with loky instead of multiprocessing."""
    supports_retrieve_callback = True
    supports_inner_max_num_threads = True

    def configure(self, n_jobs=1, parallel=None, prefer=None, require=None, idle_worker_timeout=300, **memmappingexecutor_args):
        """Build a process executor and return the number of workers"""
        n_jobs = self.effective_n_jobs(n_jobs)
        if n_jobs == 1:
            raise FallbackToBackend(SequentialBackend(nesting_level=self.nesting_level))
        self._workers = get_memmapping_executor(n_jobs, timeout=idle_worker_timeout, env=self._prepare_worker_env(n_jobs=n_jobs), context_id=parallel._id, **memmappingexecutor_args)
        self.parallel = parallel
        return n_jobs

    def effective_n_jobs(self, n_jobs):
        """Determine the number of jobs which are going to run in parallel"""
        if n_jobs == 0:
            raise ValueError('n_jobs == 0 in Parallel has no meaning')
        elif mp is None or n_jobs is None:
            return 1
        elif mp.current_process().daemon:
            if n_jobs != 1:
                if inside_dask_worker():
                    msg = "Inside a Dask worker with daemon=True, setting n_jobs=1.\nPossible work-arounds:\n- dask.config.set({'distributed.worker.daemon': False})\n- set the environment variable DASK_DISTRIBUTED__WORKER__DAEMON=False\nbefore creating your Dask cluster."
                else:
                    msg = 'Loky-backed parallel loops cannot be called in a multiprocessing, setting n_jobs=1'
                warnings.warn(msg, stacklevel=3)
            return 1
        elif not (self.in_main_thread() or self.nesting_level == 0):
            if n_jobs != 1:
                warnings.warn('Loky-backed parallel loops cannot be nested below threads, setting n_jobs=1', stacklevel=3)
            return 1
        elif n_jobs < 0:
            n_jobs = max(cpu_count() + 1 + n_jobs, 1)
        return n_jobs

    def apply_async(self, func, callback=None):
        """Schedule a func to be run"""
        future = self._workers.submit(func)
        if callback is not None:
            future.add_done_callback(callback)
        return future

    def retrieve_result_callback(self, out):
        try:
            return out.result()
        except ShutdownExecutorError:
            raise RuntimeError("The executor underlying Parallel has been shutdown. This is likely due to the garbage collection of a previous generator from a call to Parallel with return_as='generator'. Make sure the generator is not garbage collected when submitting a new job or that it is first properly exhausted.")

    def terminate(self):
        if self._workers is not None:
            self._workers._temp_folder_manager._clean_temporary_resources(context_id=self.parallel._id, force=False)
            self._workers = None
        self.reset_batch_stats()

    def abort_everything(self, ensure_ready=True):
        """Shutdown the workers and restart a new one with the same parameters
        """
        self._workers.terminate(kill_workers=True)
        self._workers = None
        if ensure_ready:
            self.configure(n_jobs=self.parallel.n_jobs, parallel=self.parallel)
import gc
import os
import warnings
import threading
import contextlib
from abc import ABCMeta, abstractmethod
from ._multiprocessing_helpers import mp
def _prepare_worker_env(self, n_jobs):
    """Return environment variables limiting threadpools in external libs.

        This function return a dict containing environment variables to pass
        when creating a pool of process. These environment variables limit the
        number of threads to `n_threads` for OpenMP, MKL, Accelerated and
        OpenBLAS libraries in the child processes.
        """
    explicit_n_threads = self.inner_max_num_threads
    default_n_threads = str(max(cpu_count() // n_jobs, 1))
    env = {}
    for var in self.MAX_NUM_THREADS_VARS:
        if explicit_n_threads is None:
            var_value = os.environ.get(var, None)
            if var_value is None:
                var_value = default_n_threads
        else:
            var_value = str(explicit_n_threads)
        env[var] = var_value
    if self.TBB_ENABLE_IPC_VAR not in os.environ:
        env[self.TBB_ENABLE_IPC_VAR] = '1'
    return env
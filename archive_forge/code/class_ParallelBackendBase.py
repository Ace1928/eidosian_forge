import gc
import os
import warnings
import threading
import contextlib
from abc import ABCMeta, abstractmethod
from ._multiprocessing_helpers import mp
class ParallelBackendBase(metaclass=ABCMeta):
    """Helper abc which defines all methods a ParallelBackend must implement"""
    supports_inner_max_num_threads = False
    supports_retrieve_callback = False
    default_n_jobs = 1

    @property
    def supports_return_generator(self):
        return self.supports_retrieve_callback

    @property
    def supports_timeout(self):
        return self.supports_retrieve_callback
    nesting_level = None

    def __init__(self, nesting_level=None, inner_max_num_threads=None, **kwargs):
        super().__init__(**kwargs)
        self.nesting_level = nesting_level
        self.inner_max_num_threads = inner_max_num_threads
    MAX_NUM_THREADS_VARS = ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMBA_NUM_THREADS', 'NUMEXPR_NUM_THREADS']
    TBB_ENABLE_IPC_VAR = 'ENABLE_IPC'

    @abstractmethod
    def effective_n_jobs(self, n_jobs):
        """Determine the number of jobs that can actually run in parallel

        n_jobs is the number of workers requested by the callers. Passing
        n_jobs=-1 means requesting all available workers for instance matching
        the number of CPU cores on the worker host(s).

        This method should return a guesstimate of the number of workers that
        can actually perform work concurrently. The primary use case is to make
        it possible for the caller to know in how many chunks to slice the
        work.

        In general working on larger data chunks is more efficient (less
        scheduling overhead and better use of CPU cache prefetching heuristics)
        as long as all the workers have enough work to do.
        """

    @abstractmethod
    def apply_async(self, func, callback=None):
        """Schedule a func to be run"""

    def retrieve_result_callback(self, out):
        """Called within the callback function passed in apply_async.

        The argument of this function is the argument given to a callback in
        the considered backend. It is supposed to return the outcome of a task
        if it succeeded or raise the exception if it failed.
        """

    def configure(self, n_jobs=1, parallel=None, prefer=None, require=None, **backend_args):
        """Reconfigure the backend and return the number of workers.

        This makes it possible to reuse an existing backend instance for
        successive independent calls to Parallel with different parameters.
        """
        self.parallel = parallel
        return self.effective_n_jobs(n_jobs)

    def start_call(self):
        """Call-back method called at the beginning of a Parallel call"""

    def stop_call(self):
        """Call-back method called at the end of a Parallel call"""

    def terminate(self):
        """Shutdown the workers and free the shared memory."""

    def compute_batch_size(self):
        """Determine the optimal batch size"""
        return 1

    def batch_completed(self, batch_size, duration):
        """Callback indicate how long it took to run a batch"""

    def get_exceptions(self):
        """List of exception types to be captured."""
        return []

    def abort_everything(self, ensure_ready=True):
        """Abort any running tasks

        This is called when an exception has been raised when executing a task
        and all the remaining tasks will be ignored and can therefore be
        aborted to spare computation resources.

        If ensure_ready is True, the backend should be left in an operating
        state as future tasks might be re-submitted via that same backend
        instance.

        If ensure_ready is False, the implementer of this method can decide
        to leave the backend in a closed / terminated state as no new task
        are expected to be submitted to this backend.

        Setting ensure_ready to False is an optimization that can be leveraged
        when aborting tasks via killing processes from a local process pool
        managed by the backend it-self: if we expect no new tasks, there is no
        point in re-creating new workers.
        """
        pass

    def get_nested_backend(self):
        """Backend instance to be used by nested Parallel calls.

        By default a thread-based backend is used for the first level of
        nesting. Beyond, switch to sequential backend to avoid spawning too
        many threads on the host.
        """
        nesting_level = getattr(self, 'nesting_level', 0) + 1
        if nesting_level > 1:
            return (SequentialBackend(nesting_level=nesting_level), None)
        else:
            return (ThreadingBackend(nesting_level=nesting_level), None)

    @contextlib.contextmanager
    def retrieval_context(self):
        """Context manager to manage an execution context.

        Calls to Parallel.retrieve will be made inside this context.

        By default, this does nothing. It may be useful for subclasses to
        handle nested parallelism. In particular, it may be required to avoid
        deadlocks if a backend manages a fixed number of workers, when those
        workers may be asked to do nested Parallel calls. Without
        'retrieval_context' this could lead to deadlock, as all the workers
        managed by the backend may be "busy" waiting for the nested parallel
        calls to finish, but the backend has no free workers to execute those
        tasks.
        """
        yield

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

    @staticmethod
    def in_main_thread():
        return isinstance(threading.current_thread(), threading._MainThread)
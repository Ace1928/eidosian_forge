import time
import warnings
import threading
import multiprocessing as mp
from .process_executor import ProcessPoolExecutor, EXTRA_QUEUED_CALLS
from .backend.context import cpu_count
from .backend import get_context
class _ReusablePoolExecutor(ProcessPoolExecutor):

    def __init__(self, submit_resize_lock, max_workers=None, context=None, timeout=None, executor_id=0, job_reducers=None, result_reducers=None, initializer=None, initargs=(), env=None):
        super().__init__(max_workers=max_workers, context=context, timeout=timeout, job_reducers=job_reducers, result_reducers=result_reducers, initializer=initializer, initargs=initargs, env=env)
        self.executor_id = executor_id
        self._submit_resize_lock = submit_resize_lock

    @classmethod
    def get_reusable_executor(cls, max_workers=None, context=None, timeout=10, kill_workers=False, reuse='auto', job_reducers=None, result_reducers=None, initializer=None, initargs=(), env=None):
        with _executor_lock:
            global _executor, _executor_kwargs
            executor = _executor
            if max_workers is None:
                if reuse is True and executor is not None:
                    max_workers = executor._max_workers
                else:
                    max_workers = cpu_count()
            elif max_workers <= 0:
                raise ValueError(f'max_workers must be greater than 0, got {max_workers}.')
            if isinstance(context, str):
                context = get_context(context)
            if context is not None and context.get_start_method() == 'fork':
                raise ValueError("Cannot use reusable executor with the 'fork' context")
            kwargs = dict(context=context, timeout=timeout, job_reducers=job_reducers, result_reducers=result_reducers, initializer=initializer, initargs=initargs, env=env)
            if executor is None:
                is_reused = False
                mp.util.debug(f'Create a executor with max_workers={max_workers}.')
                executor_id = _get_next_executor_id()
                _executor_kwargs = kwargs
                _executor = executor = cls(_executor_lock, max_workers=max_workers, executor_id=executor_id, **kwargs)
            else:
                if reuse == 'auto':
                    reuse = kwargs == _executor_kwargs
                if executor._flags.broken or executor._flags.shutdown or (not reuse):
                    if executor._flags.broken:
                        reason = 'broken'
                    elif executor._flags.shutdown:
                        reason = 'shutdown'
                    else:
                        reason = 'arguments have changed'
                    mp.util.debug(f'Creating a new executor with max_workers={max_workers} as the previous instance cannot be reused ({reason}).')
                    executor.shutdown(wait=True, kill_workers=kill_workers)
                    _executor = executor = _executor_kwargs = None
                    return cls.get_reusable_executor(max_workers=max_workers, **kwargs)
                else:
                    mp.util.debug(f'Reusing existing executor with max_workers={executor._max_workers}.')
                    is_reused = True
                    executor._resize(max_workers)
        return (executor, is_reused)

    def submit(self, fn, *args, **kwargs):
        with self._submit_resize_lock:
            return super().submit(fn, *args, **kwargs)

    def _resize(self, max_workers):
        with self._submit_resize_lock:
            if max_workers is None:
                raise ValueError('Trying to resize with max_workers=None')
            elif max_workers == self._max_workers:
                return
            if self._executor_manager_thread is None:
                self._max_workers = max_workers
                return
            self._wait_job_completion()
            with self._processes_management_lock:
                processes = list(self._processes.values())
                nb_children_alive = sum((p.is_alive() for p in processes))
                self._max_workers = max_workers
                for _ in range(max_workers, nb_children_alive):
                    self._call_queue.put(None)
            while len(self._processes) > max_workers and (not self._flags.broken):
                time.sleep(0.001)
            self._adjust_process_count()
            processes = list(self._processes.values())
            while not all((p.is_alive() for p in processes)):
                time.sleep(0.001)

    def _wait_job_completion(self):
        """Wait for the cache to be empty before resizing the pool."""
        if self._pending_work_items:
            warnings.warn('Trying to resize an executor with running jobs: waiting for jobs completion before resizing.', UserWarning)
            mp.util.debug(f'Executor {self.executor_id} waiting for jobs completion before resizing')
        while self._pending_work_items:
            time.sleep(0.001)

    def _setup_queues(self, job_reducers, result_reducers):
        queue_size = 2 * cpu_count() + EXTRA_QUEUED_CALLS
        super()._setup_queues(job_reducers, result_reducers, queue_size=queue_size)
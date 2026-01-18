import abc
import futurist
from taskflow import task as ta
from taskflow.types import failure
from taskflow.types import notifier
class ParallelTaskExecutor(TaskExecutor):
    """Executes tasks in parallel.

    Submits tasks to an executor which should provide an interface similar
    to concurrent.Futures.Executor.
    """
    constructor_options = [('max_workers', lambda v: v if v is None else int(v))]
    '\n    Optional constructor keyword arguments this executor supports. These will\n    typically be passed via engine options (by a engine user) and converted\n    into the correct type before being sent into this\n    classes ``__init__`` method.\n    '

    def __init__(self, executor=None, max_workers=None):
        self._executor = executor
        self._max_workers = max_workers
        self._own_executor = executor is None

    @abc.abstractmethod
    def _create_executor(self, max_workers=None):
        """Called when an executor has not been provided to make one."""

    def _submit_task(self, func, task, *args, **kwargs):
        fut = self._executor.submit(func, task, *args, **kwargs)
        fut.atom = task
        return fut

    def execute_task(self, task, task_uuid, arguments, progress_callback=None):
        return self._submit_task(_execute_task, task, arguments, progress_callback=progress_callback)

    def revert_task(self, task, task_uuid, arguments, result, failures, progress_callback=None):
        return self._submit_task(_revert_task, task, arguments, result, failures, progress_callback=progress_callback)

    def start(self):
        if self._own_executor:
            self._executor = self._create_executor(max_workers=self._max_workers)

    def stop(self):
        if self._own_executor:
            self._executor.shutdown(wait=True)
            self._executor = None
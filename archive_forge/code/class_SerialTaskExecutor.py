import abc
import futurist
from taskflow import task as ta
from taskflow.types import failure
from taskflow.types import notifier
class SerialTaskExecutor(TaskExecutor):
    """Executes tasks one after another."""

    def __init__(self):
        self._executor = futurist.SynchronousExecutor()

    def start(self):
        self._executor.restart()

    def stop(self):
        self._executor.shutdown()

    def execute_task(self, task, task_uuid, arguments, progress_callback=None):
        fut = self._executor.submit(_execute_task, task, arguments, progress_callback=progress_callback)
        fut.atom = task
        return fut

    def revert_task(self, task, task_uuid, arguments, result, failures, progress_callback=None):
        fut = self._executor.submit(_revert_task, task, arguments, result, failures, progress_callback=progress_callback)
        fut.atom = task
        return fut
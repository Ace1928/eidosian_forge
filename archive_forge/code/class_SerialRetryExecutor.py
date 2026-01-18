import abc
import futurist
from taskflow import task as ta
from taskflow.types import failure
from taskflow.types import notifier
class SerialRetryExecutor(object):
    """Executes and reverts retries."""

    def __init__(self):
        self._executor = futurist.SynchronousExecutor()

    def start(self):
        """Prepare to execute retries."""
        self._executor.restart()

    def stop(self):
        """Finalize retry executor."""
        self._executor.shutdown()

    def execute_retry(self, retry, arguments):
        """Schedules retry execution."""
        fut = self._executor.submit(_execute_retry, retry, arguments)
        fut.atom = retry
        return fut

    def revert_retry(self, retry, arguments):
        """Schedules retry reversion."""
        fut = self._executor.submit(_revert_retry, retry, arguments)
        fut.atom = retry
        return fut
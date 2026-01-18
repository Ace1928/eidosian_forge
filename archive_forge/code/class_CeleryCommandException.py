import numbers
from billiard.exceptions import SoftTimeLimitExceeded, Terminated, TimeLimitExceeded, WorkerLostError
from click import ClickException
from kombu.exceptions import OperationalError
from celery.utils.serialization import get_pickleable_exception
class CeleryCommandException(ClickException):
    """A general command exception which stores an exit code."""

    def __init__(self, message, exit_code):
        super().__init__(message=message)
        self.exit_code = exit_code
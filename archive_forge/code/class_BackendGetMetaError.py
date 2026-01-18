import numbers
from billiard.exceptions import SoftTimeLimitExceeded, Terminated, TimeLimitExceeded, WorkerLostError
from click import ClickException
from kombu.exceptions import OperationalError
from celery.utils.serialization import get_pickleable_exception
class BackendGetMetaError(BackendError):
    """An issue reading from the backend."""

    def __init__(self, *args, **kwargs):
        self.task_id = kwargs.get('task_id', '')

    def __repr__(self):
        return super().__repr__() + ' task_id:' + self.task_id
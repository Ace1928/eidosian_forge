import errno
import socket
from celery import bootsteps
from celery.exceptions import WorkerLostError
from celery.utils.log import get_logger
from . import state
def _quick_drain(connection, timeout=0.1):
    try:
        connection.drain_events(timeout=timeout)
    except Exception as exc:
        exc_errno = getattr(exc, 'errno', None)
        if exc_errno is not None and exc_errno != errno.EAGAIN:
            raise
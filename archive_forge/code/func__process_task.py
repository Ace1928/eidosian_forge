import os
import sys
from datetime import datetime, timezone
from billiard import cpu_count
from kombu.utils.compat import detect_environment
from celery import bootsteps
from celery import concurrency as _concurrency
from celery import signals
from celery.bootsteps import RUN, TERMINATE
from celery.exceptions import ImproperlyConfigured, TaskRevokedError, WorkerTerminate
from celery.platforms import EX_FAILURE, create_pidlock
from celery.utils.imports import reload_from_cwd
from celery.utils.log import mlevel
from celery.utils.log import worker_logger as logger
from celery.utils.nodenames import default_nodename, worker_direct
from celery.utils.text import str_to_list
from celery.utils.threads import default_socket_timeout
from . import state
def _process_task(self, req):
    """Process task by sending it to the pool of workers."""
    try:
        req.execute_using_pool(self.pool)
    except TaskRevokedError:
        try:
            self._quick_release()
        except AttributeError:
            pass
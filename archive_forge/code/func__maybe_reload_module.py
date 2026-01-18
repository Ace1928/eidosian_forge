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
def _maybe_reload_module(self, module, force_reload=False, reloader=None):
    if module not in sys.modules:
        logger.debug('importing module %s', module)
        return self.app.loader.import_from_cwd(module)
    elif force_reload:
        logger.debug('reloading module %s', module)
        return reload_from_cwd(sys.modules[module], reloader)
import abc
import contextlib
import functools
import itertools
import threading
from oslo_utils import excutils
from oslo_utils import timeutils
from taskflow.conductors import base
from taskflow import exceptions as excp
from taskflow.listeners import logging as logging_listener
from taskflow import logging
from taskflow import states
from taskflow.types import timing as tt
from taskflow.utils import iter_utils
from taskflow.utils import misc
def _try_finish_job(self, job, consume, trash=False):
    try:
        if consume:
            self._jobboard.consume(job, self._name)
            self._notifier.notify('job_consumed', {'job': job, 'conductor': self, 'persistence': self._persistence})
        elif trash:
            self._jobboard.trash(job, self._name)
            self._notifier.notify('job_trashed', {'job': job, 'conductor': self, 'persistence': self._persistence})
        else:
            self._jobboard.abandon(job, self._name)
            self._notifier.notify('job_abandoned', {'job': job, 'conductor': self, 'persistence': self._persistence})
    except (excp.JobFailure, excp.NotFound):
        if consume:
            self._log.warn('Failed job consumption: %s', job, exc_info=True)
        else:
            self._log.warn('Failed job abandonment: %s', job, exc_info=True)
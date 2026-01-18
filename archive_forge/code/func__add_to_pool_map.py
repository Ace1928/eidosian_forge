import sys
from time import monotonic
from greenlet import GreenletExit
from kombu.asynchronous import timer as _timer
from celery import signals
from . import base
def _add_to_pool_map(self, pid, greenlet):
    self._pool_map[pid] = greenlet
    greenlet.link(TaskPool._cleanup_after_job_finish, self._pool_map, pid)
import sys
from time import monotonic
from greenlet import GreenletExit
from kombu.asynchronous import timer as _timer
from celery import signals
from . import base
@staticmethod
def _cleanup_after_job_finish(greenlet, pool_map, pid):
    del pool_map[pid]
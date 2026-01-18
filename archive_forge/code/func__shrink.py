import os
import threading
from time import monotonic, sleep
from kombu.asynchronous.semaphore import DummyLock
from celery import bootsteps
from celery.utils.log import get_logger
from celery.utils.threads import bgThread
from . import state
from .components import Pool
def _shrink(self, n):
    info('Scaling down %s processes.', n)
    try:
        self.pool.shrink(n)
    except ValueError:
        debug("Autoscaler won't scale down: all processes busy.")
    except Exception as exc:
        error('Autoscaler: scale_down: %r', exc, exc_info=True)
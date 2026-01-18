import os
import threading
from time import monotonic, sleep
from kombu.asynchronous.semaphore import DummyLock
from celery import bootsteps
from celery.utils.log import get_logger
from celery.utils.threads import bgThread
from . import state
from .components import Pool
def _maybe_scale(self, req=None):
    procs = self.processes
    cur = min(self.qty, self.max_concurrency)
    if cur > procs:
        self.scale_up(cur - procs)
        return True
    cur = max(self.qty, self.min_concurrency)
    if cur < procs:
        self.scale_down(procs - cur)
        return True
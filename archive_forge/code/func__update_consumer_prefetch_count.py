import os
import threading
from time import monotonic, sleep
from kombu.asynchronous.semaphore import DummyLock
from celery import bootsteps
from celery.utils.log import get_logger
from celery.utils.threads import bgThread
from . import state
from .components import Pool
def _update_consumer_prefetch_count(self, new_max):
    diff = new_max - self.max_concurrency
    if diff:
        self.worker.consumer._update_prefetch_count(diff)
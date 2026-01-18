import os
import threading
from time import monotonic, sleep
from kombu.asynchronous.semaphore import DummyLock
from celery import bootsteps
from celery.utils.log import get_logger
from celery.utils.threads import bgThread
from . import state
from .components import Pool
class WorkerComponent(bootsteps.StartStopStep):
    """Bootstep that starts the autoscaler thread/timer in the worker."""
    label = 'Autoscaler'
    conditional = True
    requires = (Pool,)

    def __init__(self, w, **kwargs):
        self.enabled = w.autoscale
        w.autoscaler = None

    def create(self, w):
        scaler = w.autoscaler = self.instantiate(w.autoscaler_cls, w.pool, w.max_concurrency, w.min_concurrency, worker=w, mutex=DummyLock() if w.use_eventloop else None)
        return scaler if not w.use_eventloop else None

    def register_with_event_loop(self, w, hub):
        w.consumer.on_task_message.add(w.autoscaler.maybe_scale)
        hub.call_repeatedly(w.autoscaler.keepalive, w.autoscaler.maybe_scale)

    def info(self, w):
        """Return `Autoscaler` info."""
        return {'autoscaler': w.autoscaler.info()}
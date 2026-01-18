import atexit
import warnings
from kombu.asynchronous import Hub as _Hub
from kombu.asynchronous import get_event_loop, set_event_loop
from kombu.asynchronous.semaphore import DummyLock, LaxBoundedSemaphore
from kombu.asynchronous.timer import Timer as _Timer
from celery import bootsteps
from celery._state import _set_task_join_will_block
from celery.exceptions import ImproperlyConfigured
from celery.platforms import IS_WINDOWS
from celery.utils.log import worker_logger as logger
def _patch_thread_primitives(self, w):
    w.app.clock.mutex = DummyLock()
    try:
        from billiard import pool
    except ImportError:
        pass
    else:
        pool.Lock = DummyLock
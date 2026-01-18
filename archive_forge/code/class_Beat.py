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
class Beat(bootsteps.StartStopStep):
    """Step used to embed a beat process.

    Enabled when the ``beat`` argument is set.
    """
    label = 'Beat'
    conditional = True

    def __init__(self, w, beat=False, **kwargs):
        self.enabled = w.beat = beat
        w.beat = None
        super().__init__(w, beat=beat, **kwargs)

    def create(self, w):
        from celery.beat import EmbeddedService
        if w.pool_cls.__module__.endswith(('gevent', 'eventlet')):
            raise ImproperlyConfigured(ERR_B_GREEN)
        b = w.beat = EmbeddedService(w.app, schedule_filename=w.schedule_filename, scheduler_cls=w.scheduler)
        return b
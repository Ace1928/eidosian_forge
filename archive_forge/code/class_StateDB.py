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
class StateDB(bootsteps.Step):
    """Bootstep that sets up between-restart state database file."""

    def __init__(self, w, **kwargs):
        self.enabled = w.statedb
        w._persistence = None
        super().__init__(w, **kwargs)

    def create(self, w):
        w._persistence = w.state.Persistent(w.state, w.statedb, w.app.clock)
        atexit.register(w._persistence.save)
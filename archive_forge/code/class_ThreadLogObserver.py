import threading
import weakref
import warnings
from inspect import iscoroutinefunction
from functools import wraps
from queue import SimpleQueue
from twisted.python import threadable
from twisted.python.runtime import platform
from twisted.python.failure import Failure
from twisted.python.log import PythonLoggingObserver, err
from twisted.internet.defer import maybeDeferred, ensureDeferred
from twisted.internet.task import LoopingCall
import wrapt
from ._util import synchronized
from ._resultstore import ResultStore
class ThreadLogObserver(object):
    """
    A log observer that wraps another observer, and calls it in a thread.

    In particular, used to wrap PythonLoggingObserver, so that blocking
    logging.py Handlers don't block the event loop.
    """

    def __init__(self, observer):
        self._observer = observer
        self._queue = SimpleQueue()
        self._thread = threading.Thread(target=self._reader, name='CrochetLogWriter')
        self._thread.start()

    def _reader(self):
        """
        Runs in a thread, reads messages from a queue and writes them to
        the wrapped observer.
        """
        while True:
            msg = self._queue.get()
            if msg is _STOP:
                return
            try:
                self._observer(msg)
            except Exception:
                pass

    def stop(self):
        """
        Stop the thread.
        """
        self._queue.put(_STOP)

    def __call__(self, msg):
        """
        A log observer that writes to a queue.
        """
        self._queue.put(msg)
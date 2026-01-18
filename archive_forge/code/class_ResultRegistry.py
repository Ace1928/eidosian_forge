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
class ResultRegistry(object):
    """
    Keep track of EventualResults.

    Once the reactor has shutdown:

    1. Registering new EventualResult instances is an error, since no results
       will ever become available.
    2. Already registered EventualResult instances are "fired" with a
       ReactorStopped exception to unblock any remaining EventualResult.wait()
       calls.
    """

    def __init__(self):
        self._results = weakref.WeakSet()
        self._stopped = False
        self._lock = threading.Lock()

    @synchronized
    def register(self, result):
        """
        Register an EventualResult.

        May be called in any thread.
        """
        if self._stopped:
            raise ReactorStopped()
        self._results.add(result)

    @synchronized
    def stop(self):
        """
        Indicate no more results will get pushed into EventualResults, since
        the reactor has stopped.

        This should be called in the reactor thread.
        """
        self._stopped = True
        for result in self._results:
            result._set_result(Failure(ReactorStopped()))
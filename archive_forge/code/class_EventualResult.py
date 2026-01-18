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
class EventualResult(object):
    """
    A blocking interface to Deferred results.

    This allows you to access results from Twisted operations that may not be
    available immediately, using the wait() method.

    In general you should not create these directly; instead use functions
    decorated with @run_in_reactor.
    """

    def __init__(self, deferred, _reactor):
        """
        The deferred parameter should be a Deferred or None indicating
        _connect_deferred will be called separately later.
        """
        self._deferred = deferred
        self._reactor = _reactor
        self._value = None
        self._result_retrieved = False
        self._result_set = threading.Event()
        if deferred is not None:
            self._connect_deferred(deferred)

    def _connect_deferred(self, deferred):
        """
        Hook up the Deferred that that this will be the result of.

        Should only be run in Twisted thread, and only called once.
        """
        self._deferred = deferred

        def put(result, eventual=weakref.ref(self)):
            eventual = eventual()
            if eventual:
                eventual._set_result(result)
            else:
                err(result, 'Unhandled error in EventualResult')
        deferred.addBoth(put)

    def _set_result(self, result):
        """
        Set the result of the EventualResult, if not already set.

        This can only happen in the reactor thread, either as a result of
        Deferred firing, or as a result of ResultRegistry.stop(). So, no need
        for thread-safety.
        """
        if self._result_set.isSet():
            return
        self._value = result
        self._result_set.set()

    def __del__(self):
        if self._result_retrieved or not self._result_set.isSet():
            return
        if isinstance(self._value, Failure):
            err(self._value, 'Unhandled error in EventualResult')

    def cancel(self):
        """
        Try to cancel the operation by cancelling the underlying Deferred.

        Cancellation of the operation may or may not happen depending on
        underlying cancellation support and whether the operation has already
        finished. In any case, however, the underlying Deferred will be fired.

        Multiple calls will have no additional effect.
        """
        self._reactor.callFromThread(lambda: self._deferred.cancel())

    def _result(self, timeout):
        """
        Return the result, if available.

        It may take an unknown amount of time to return the result, so a
        timeout option is provided. If the given number of seconds pass with
        no result, a TimeoutError will be thrown.

        If a previous call timed out, additional calls to this function will
        still wait for a result and return it if available. If a result was
        returned on one call, additional calls will return/raise the same
        result.
        """
        self._result_set.wait(timeout)
        if not self._result_set.is_set():
            raise TimeoutError()
        self._result_retrieved = True
        return self._value

    def wait(self, timeout):
        """
        Return the result, or throw the exception if result is a failure.

        It may take an unknown amount of time to return the result, so a
        timeout option is provided. If the given number of seconds pass with
        no result, a TimeoutError will be thrown.

        If a previous call timed out, additional calls to this function will
        still wait for a result and return it if available. If a result was
        returned or raised on one call, additional calls will return/raise the
        same result.
        """
        if threadable.isInIOThread():
            raise RuntimeError('EventualResult.wait() must not be run in the reactor thread.')
        result = self._result(timeout)
        if isinstance(result, Failure):
            result.raiseException()
        return result

    def stash(self):
        """
        Store the EventualResult in memory for later retrieval.

        Returns a integer uid which can be passed to crochet.retrieve_result()
        to retrieve the instance later on.
        """
        return _store.store(self)

    def original_failure(self):
        """
        Return the underlying Failure object, if the result is an error.

        If no result is yet available, or the result was not an error, None is
        returned.

        This method is useful if you want to get the original traceback for an
        error result.
        """
        try:
            result = self._result(0.0)
        except TimeoutError:
            return None
        if isinstance(result, Failure):
            return result
        else:
            return None
import functools
import queue
import threading
from concurrent import futures as _futures
from concurrent.futures import process as _process
from futurist import _green
from futurist import _thread
from futurist import _utils
class ExecutorStatistics(object):
    """Holds *immutable* information about a executors executions."""
    __slots__ = ['_failures', '_executed', '_runtime', '_cancelled']
    _REPR_MSG_TPL = '<ExecutorStatistics object at 0x%(ident)x (failures=%(failures)s, executed=%(executed)s, runtime=%(runtime)0.2f, cancelled=%(cancelled)s)>'

    def __init__(self, failures=0, executed=0, runtime=0.0, cancelled=0):
        self._failures = failures
        self._executed = executed
        self._runtime = runtime
        self._cancelled = cancelled

    @property
    def failures(self):
        """How many submissions ended up raising exceptions.

        :returns: how many submissions ended up raising exceptions
        :rtype: number
        """
        return self._failures

    @property
    def executed(self):
        """How many submissions were executed (failed or not).

        :returns: how many submissions were executed
        :rtype: number
        """
        return self._executed

    @property
    def runtime(self):
        """Total runtime of all submissions executed (failed or not).

        :returns: total runtime of all submissions executed
        :rtype: number
        """
        return self._runtime

    @property
    def cancelled(self):
        """How many submissions were cancelled before executing.

        :returns: how many submissions were cancelled before executing
        :rtype: number
        """
        return self._cancelled

    @property
    def average_runtime(self):
        """The average runtime of all submissions executed.

        :returns: average runtime of all submissions executed
        :rtype: number
        :raises: ZeroDivisionError when no executions have occurred.
        """
        return self._runtime / self._executed

    def __repr__(self):
        return self._REPR_MSG_TPL % {'ident': id(self), 'failures': self._failures, 'executed': self._executed, 'runtime': self._runtime, 'cancelled': self._cancelled}
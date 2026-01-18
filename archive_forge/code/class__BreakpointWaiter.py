from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
import threading
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.debug import errors
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import retry
import six
from six.moves import urllib
class _BreakpointWaiter(object):
    """Waits for multiple breakpoints.

  Attributes:
    _result_lock: Lock for modifications to all fields
    _done: Flag to indicate that the wait condition is satisfied and wait
        should stop even if some threads are not finished.
    _threads: The list of active threads
    _results: The set of completed breakpoints.
    _failures: All exceptions which caused any thread to stop waiting.
    _wait_all: If true, wait for all breakpoints to complete, else wait for
        any breakpoint to complete. Controls whether to set _done after any
        breakpoint completes.
    _timeout: Mazimum time (in ms) to wait for breakpoints to complete.
  """

    def __init__(self, wait_all, timeout):
        self._result_lock = threading.Lock()
        self._done = False
        self._threads = []
        self._results = {}
        self._failures = []
        self._wait_all = wait_all
        self._timeout = timeout

    def _IsComplete(self, response):
        if response.breakpoint.isFinalState:
            return True
        with self._result_lock:
            return self._done

    def _WaitForOne(self, debuggee, breakpoint_id):
        try:
            breakpoint = debuggee.WaitForBreakpoint(breakpoint_id, timeout=self._timeout, completion_test=self._IsComplete)
            if not breakpoint:
                with self._result_lock:
                    if not self._wait_all:
                        self._done = True
                return
            if breakpoint.isFinalState:
                with self._result_lock:
                    self._results[breakpoint_id] = breakpoint
                    if not self._wait_all:
                        self._done = True
        except errors.DebugError as e:
            with self._result_lock:
                self._failures.append(e)
                self._done = True

    def AddTarget(self, debuggee, breakpoint_id):
        self._threads.append(threading.Thread(target=self._WaitForOne, args=(debuggee, breakpoint_id)))

    def Wait(self):
        for t in self._threads:
            t.start()
        for t in self._threads:
            t.join()
        if self._failures:
            raise self._failures[0]
        return self._results
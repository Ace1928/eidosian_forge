from __future__ import annotations
import collections.abc as c
import contextlib
import functools
import sys
import threading
import queue
import typing as t
class WrappedThread(threading.Thread):
    """Wrapper around Thread which captures results and exceptions."""

    def __init__(self, action: c.Callable[[], t.Any]) -> None:
        super().__init__()
        self._result: queue.Queue[t.Any] = queue.Queue()
        self.action = action
        self.result = None

    def run(self) -> None:
        """
        Run action and capture results or exception.
        Do not override. Do not call directly. Executed by the start() method.
        """
        try:
            self._result.put((self.action(), None))
        except:
            self._result.put((None, sys.exc_info()))

    def wait_for_result(self) -> t.Any:
        """Wait for thread to exit and return the result or raise an exception."""
        result, exception = self._result.get()
        if exception:
            raise exception[1].with_traceback(exception[2])
        self.result = result
        return result
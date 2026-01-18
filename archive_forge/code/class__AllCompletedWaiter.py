import collections
import logging
import threading
import time
import types
class _AllCompletedWaiter(_Waiter):
    """Used by wait(return_when=FIRST_EXCEPTION and ALL_COMPLETED)."""

    def __init__(self, num_pending_calls, stop_on_exception):
        self.num_pending_calls = num_pending_calls
        self.stop_on_exception = stop_on_exception
        self.lock = threading.Lock()
        super().__init__()

    def _decrement_pending_calls(self):
        with self.lock:
            self.num_pending_calls -= 1
            if not self.num_pending_calls:
                self.event.set()

    def add_result(self, future):
        super().add_result(future)
        self._decrement_pending_calls()

    def add_exception(self, future):
        super().add_exception(future)
        if self.stop_on_exception:
            self.event.set()
        else:
            self._decrement_pending_calls()

    def add_cancelled(self, future):
        super().add_cancelled(future)
        self._decrement_pending_calls()
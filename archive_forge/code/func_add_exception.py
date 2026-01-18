import collections
import logging
import threading
import time
import types
def add_exception(self, future):
    super().add_exception(future)
    if self.stop_on_exception:
        self.event.set()
    else:
        self._decrement_pending_calls()
import collections
import logging
import threading
import time
import types
def _decrement_pending_calls(self):
    with self.lock:
        self.num_pending_calls -= 1
        if not self.num_pending_calls:
            self.event.set()
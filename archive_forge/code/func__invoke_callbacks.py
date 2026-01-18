import collections
import logging
import threading
import time
import types
def _invoke_callbacks(self):
    for callback in self._done_callbacks:
        try:
            callback(self)
        except Exception:
            LOGGER.exception('exception calling callback for %r', self)
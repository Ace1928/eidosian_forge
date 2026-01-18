import collections
import datetime
import logging
import queue as queue_module
import threading
import time
from google.api_core import exceptions
def _recoverable(self, method, *args, **kwargs):
    """Wraps a method to recover the stream and retry on error.

        If a retryable error occurs while making the call, then the stream will
        be re-opened and the method will be retried. This happens indefinitely
        so long as the error is a retryable one. If an error occurs while
        re-opening the stream, then this method will raise immediately and
        trigger finalization of this object.

        Args:
            method (Callable[..., Any]): The method to call.
            args: The args to pass to the method.
            kwargs: The kwargs to pass to the method.
        """
    while True:
        try:
            return method(*args, **kwargs)
        except Exception as exc:
            with self._operational_lock:
                _LOGGER.debug('Call to retryable %r caused %s.', method, exc)
                if self._should_terminate(exc):
                    self.close()
                    _LOGGER.debug('Terminating %r due to %s.', method, exc)
                    self._finalize(exc)
                    break
                if not self._should_recover(exc):
                    self.close()
                    _LOGGER.debug('Not retrying %r due to %s.', method, exc)
                    self._finalize(exc)
                    raise exc
                _LOGGER.debug('Re-opening stream from retryable %r.', method)
                self._reopen()
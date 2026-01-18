import datetime
import errno
import os
import os.path
import time
class RateLimiter:
    """Helper class for rate-limiting using a fixed minimum interval."""

    def __init__(self, interval_secs):
        """Constructs a RateLimiter that permits a tick() every
        `interval_secs`."""
        self._time = time
        self._interval_secs = interval_secs
        self._last_called_secs = 0

    def tick(self):
        """Blocks until it has been at least `interval_secs` since last
        tick()."""
        wait_secs = self._last_called_secs + self._interval_secs - self._time.time()
        if wait_secs > 0:
            self._time.sleep(wait_secs)
        self._last_called_secs = self._time.time()
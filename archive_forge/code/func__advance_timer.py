import queue
import threading
import time
def _advance_timer(self):
    """Calculate the time when it's ok to run a command again.

        This runs inside of the mutex, serializing the calculation
        of when it's ok to run again and setting _rate_last_ts to that
        new time so that the next thread to calculate when it's safe to
        run starts from the time that the current thread calculated.
        """
    self._rate_last_ts = self._rate_last_ts + self._rate_delay
    return self._rate_last_ts
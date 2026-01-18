import collections
import datetime
import logging
import queue as queue_module
import threading
import time
from google.api_core import exceptions
class _Throttle(object):
    """A context manager limiting the total entries in a sliding time window.

    If more than ``access_limit`` attempts are made to enter the context manager
    instance in the last ``time window`` interval, the exceeding requests block
    until enough time elapses.

    The context manager instances are thread-safe and can be shared between
    multiple threads. If multiple requests are blocked and waiting to enter,
    the exact order in which they are allowed to proceed is not determined.

    Example::

        max_three_per_second = _Throttle(
            access_limit=3, time_window=datetime.timedelta(seconds=1)
        )

        for i in range(5):
            with max_three_per_second as time_waited:
                print("{}: Waited {} seconds to enter".format(i, time_waited))

    Args:
        access_limit (int): the maximum number of entries allowed in the time window
        time_window (datetime.timedelta): the width of the sliding time window
    """

    def __init__(self, access_limit, time_window):
        if access_limit < 1:
            raise ValueError('access_limit argument must be positive')
        if time_window <= datetime.timedelta(0):
            raise ValueError('time_window argument must be a positive timedelta')
        self._time_window = time_window
        self._access_limit = access_limit
        self._past_entries = collections.deque(maxlen=access_limit)
        self._entry_lock = threading.Lock()

    def __enter__(self):
        with self._entry_lock:
            cutoff_time = datetime.datetime.now() - self._time_window
            while self._past_entries and self._past_entries[0] < cutoff_time:
                self._past_entries.popleft()
            if len(self._past_entries) < self._access_limit:
                self._past_entries.append(datetime.datetime.now())
                return 0.0
            to_wait = (self._past_entries[0] - cutoff_time).total_seconds()
            time.sleep(to_wait)
            self._past_entries.append(datetime.datetime.now())
            return to_wait

    def __exit__(self, *_):
        pass

    def __repr__(self):
        return '{}(access_limit={}, time_window={})'.format(self.__class__.__name__, self._access_limit, repr(self._time_window))
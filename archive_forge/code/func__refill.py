import threading
import time
from botocore.exceptions import CapacityNotAvailableError
def _refill(self):
    timestamp = self._clock.current_time()
    if self._last_timestamp is None:
        self._last_timestamp = timestamp
        return
    current_capacity = self._current_capacity
    fill_amount = (timestamp - self._last_timestamp) * self._fill_rate
    new_capacity = min(self._max_capacity, current_capacity + fill_amount)
    self._current_capacity = new_capacity
    self._last_timestamp = timestamp
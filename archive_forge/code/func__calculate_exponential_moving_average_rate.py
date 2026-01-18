import time
import threading
def _calculate_exponential_moving_average_rate(self, amt, time_at_consumption):
    new_rate = self._calculate_rate(amt, time_at_consumption)
    return self._alpha * new_rate + (1 - self._alpha) * self._current_rate
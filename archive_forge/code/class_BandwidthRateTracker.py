import time
import threading
class BandwidthRateTracker(object):

    def __init__(self, alpha=0.8):
        """Tracks the rate of bandwidth consumption

        :type a: float
        :param a: The constant to use in calculating the exponentional moving
            average of the bandwidth rate. Specifically it is used in the
            following calculation:

            current_rate = alpha * new_rate + (1 - alpha) * current_rate

            This value of this constant should be between 0 and 1.
        """
        self._alpha = alpha
        self._last_time = None
        self._current_rate = None

    @property
    def current_rate(self):
        """The current transfer rate

        :rtype: float
        :returns: The current tracked transfer rate
        """
        if self._last_time is None:
            return 0.0
        return self._current_rate

    def get_projected_rate(self, amt, time_at_consumption):
        """Get the projected rate using a provided amount and time

        :type amt: int
        :param amt: The proposed amount to consume

        :type time_at_consumption: float
        :param time_at_consumption: The proposed time to consume at

        :rtype: float
        :returns: The consumption rate if that amt and time were consumed
        """
        if self._last_time is None:
            return 0.0
        return self._calculate_exponential_moving_average_rate(amt, time_at_consumption)

    def record_consumption_rate(self, amt, time_at_consumption):
        """Record the consumption rate based off amount and time point

        :type amt: int
        :param amt: The amount that got consumed

        :type time_at_consumption: float
        :param time_at_consumption: The time at which the amount was consumed
        """
        if self._last_time is None:
            self._last_time = time_at_consumption
            self._current_rate = 0.0
            return
        self._current_rate = self._calculate_exponential_moving_average_rate(amt, time_at_consumption)
        self._last_time = time_at_consumption

    def _calculate_rate(self, amt, time_at_consumption):
        time_delta = time_at_consumption - self._last_time
        if time_delta <= 0:
            return float('inf')
        return amt / time_delta

    def _calculate_exponential_moving_average_rate(self, amt, time_at_consumption):
        new_rate = self._calculate_rate(amt, time_at_consumption)
        return self._alpha * new_rate + (1 - self._alpha) * self._current_rate
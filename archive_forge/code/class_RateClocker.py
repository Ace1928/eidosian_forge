import logging
import math
import threading
from botocore.retries import bucket, standard, throttling
class RateClocker:
    """Tracks the rate at which a client is sending a request."""
    _DEFAULT_SMOOTHING = 0.8
    _TIME_BUCKET_RANGE = 0.5

    def __init__(self, clock, smoothing=_DEFAULT_SMOOTHING, time_bucket_range=_TIME_BUCKET_RANGE):
        self._clock = clock
        self._measured_rate = 0
        self._smoothing = smoothing
        self._last_bucket = math.floor(self._clock.current_time())
        self._time_bucket_scale = 1 / self._TIME_BUCKET_RANGE
        self._count = 0
        self._lock = threading.Lock()

    def record(self, amount=1):
        with self._lock:
            t = self._clock.current_time()
            bucket = math.floor(t * self._time_bucket_scale) / self._time_bucket_scale
            self._count += amount
            if bucket > self._last_bucket:
                current_rate = self._count / float(bucket - self._last_bucket)
                self._measured_rate = current_rate * self._smoothing + self._measured_rate * (1 - self._smoothing)
                self._count = 0
                self._last_bucket = bucket
            return self._measured_rate

    @property
    def measured_rate(self):
        return self._measured_rate
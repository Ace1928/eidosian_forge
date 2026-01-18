import logging
import math
import threading
from botocore.retries import bucket, standard, throttling
class ClientRateLimiter:
    _MAX_RATE_ADJUST_SCALE = 2.0

    def __init__(self, rate_adjustor, rate_clocker, token_bucket, throttling_detector, clock):
        self._rate_adjustor = rate_adjustor
        self._rate_clocker = rate_clocker
        self._token_bucket = token_bucket
        self._throttling_detector = throttling_detector
        self._clock = clock
        self._enabled = False
        self._lock = threading.Lock()

    def on_sending_request(self, request, **kwargs):
        if self._enabled:
            self._token_bucket.acquire()

    def on_receiving_response(self, **kwargs):
        measured_rate = self._rate_clocker.record()
        timestamp = self._clock.current_time()
        with self._lock:
            if not self._throttling_detector.is_throttling_error(**kwargs):
                new_rate = self._rate_adjustor.success_received(timestamp)
            else:
                if not self._enabled:
                    rate_to_use = measured_rate
                else:
                    rate_to_use = min(measured_rate, self._token_bucket.max_rate)
                new_rate = self._rate_adjustor.error_received(rate_to_use, timestamp)
                logger.debug('Throttling response received, new send rate: %s measured rate: %s, token bucket capacity available: %s', new_rate, measured_rate, self._token_bucket.available_capacity)
                self._enabled = True
            self._token_bucket.max_rate = min(new_rate, self._MAX_RATE_ADJUST_SCALE * measured_rate)
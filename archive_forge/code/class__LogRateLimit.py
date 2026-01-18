import logging
from time import monotonic as monotonic_clock
class _LogRateLimit(logging.Filter):

    def __init__(self, burst, interval, except_level=None):
        logging.Filter.__init__(self)
        self.burst = burst
        self.interval = interval
        self.except_level = except_level
        self.logger = logging.getLogger()
        self._reset()

    def _reset(self, now=None):
        if now is None:
            now = monotonic_clock()
        self.counter = 0
        self.end_time = now + self.interval
        self.emit_warn = False

    def filter(self, record):
        if self.except_level is not None and record.levelno >= self.except_level:
            return True
        timestamp = monotonic_clock()
        if timestamp >= self.end_time:
            self._reset(timestamp)
            self.counter += 1
            return True
        self.counter += 1
        if self.counter <= self.burst:
            return True
        if self.emit_warn:
            return True
        if self.counter == self.burst + 1:
            self.emit_warn = True
            self.logger.error('Logging rate limit: drop after %s records/%s sec', self.burst, self.interval)
            self.emit_warn = False
        return False
import logging
from time import monotonic as monotonic_clock
class RateLimitLogger(install_filter.logger_class):

    def __init__(self, *args, **kw):
        logging.Logger.__init__(self, *args, **kw)
        self.addFilter(log_filter)
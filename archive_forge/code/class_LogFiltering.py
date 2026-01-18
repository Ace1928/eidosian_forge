import logging
class LogFiltering:

    def __init__(self, logger, filter):
        self.logger = logger
        self.filter = filter

    def __enter__(self):
        self.logger.addFilter(self.filter)

    def __exit__(self, *args, **kwargs):
        self.logger.removeFilter(self.filter)
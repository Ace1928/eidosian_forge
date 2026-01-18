import sys
import logging
import timeit
from functools import wraps
from collections.abc import Mapping, Callable
import warnings
from logging import PercentStyle
class CapturingLogHandler(logging.Handler):

    def __init__(self, logger, level):
        super(CapturingLogHandler, self).__init__(level=level)
        self.records = []
        if isinstance(logger, str):
            self.logger = logging.getLogger(logger)
        else:
            self.logger = logger

    def __enter__(self):
        self.original_disabled = self.logger.disabled
        self.original_level = self.logger.level
        self.original_propagate = self.logger.propagate
        self.logger.addHandler(self)
        self.logger.setLevel(self.level)
        self.logger.disabled = False
        self.logger.propagate = False
        return self

    def __exit__(self, type, value, traceback):
        self.logger.removeHandler(self)
        self.logger.setLevel(self.original_level)
        self.logger.disabled = self.original_disabled
        self.logger.propagate = self.original_propagate
        return self

    def emit(self, record):
        self.records.append(record)

    def assertRegex(self, regexp, msg=None):
        import re
        pattern = re.compile(regexp)
        for r in self.records:
            if pattern.search(r.getMessage()):
                return True
        if msg is None:
            msg = "Pattern '%s' not found in logger records" % regexp
        assert 0, msg
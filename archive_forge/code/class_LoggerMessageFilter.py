import contextlib
import logging
import logging.config
import re
import sys
class LoggerMessageFilter(logging.Filter):

    def __init__(self, module: str, filter_regex: re.Pattern):
        super().__init__()
        self._pattern = filter_regex
        self._module = module

    def filter(self, record):
        if record.name == self._module and self._pattern.search(record.msg):
            return False
        return True
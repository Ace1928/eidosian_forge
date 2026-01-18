import logging
import os
from logging import Filter
class LocalProcessFilter(Filter):
    """
    Filters logs not originating from the current executing Python process ID.
    """

    def __init__(self):
        super().__init__()
        self._pid = os.getpid()

    def filter(self, record):
        if record.process == self._pid:
            return True
        return False
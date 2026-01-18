import logging
import os
from logging import Filter
class DebugOnlyFilter(Filter):
    """
    Filters logs that are less verbose than the DEBUG level (CRITICAL, ERROR, WARN & INFO).
    """

    def filter(self, record):
        super().__init__()
        if record.levelno > logging.DEBUG:
            return False
        return True
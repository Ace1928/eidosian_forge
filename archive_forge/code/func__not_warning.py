from __future__ import annotations
import logging
import sys
def _not_warning(record):
    return record.levelno < logging.WARNING
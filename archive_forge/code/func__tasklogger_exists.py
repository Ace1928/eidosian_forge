from deprecated.sphinx import deprecated
import contextlib
import logging
import time
import sys
def _tasklogger_exists(logger):
    """Check if a `logging.Logger` already has an associated TaskLogger"""
    return hasattr(logger, 'tasklogger')
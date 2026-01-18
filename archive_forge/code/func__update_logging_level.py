from both of those two places to another location.
import errno
import logging
import os
import sys
import time
from io import StringIO
import breezy
from .lazy_import import lazy_import
from breezy import (
from . import errors
def _update_logging_level(quiet=True):
    """Hide INFO messages if quiet."""
    if quiet:
        _brz_logger.setLevel(logging.WARNING)
    else:
        _brz_logger.setLevel(logging.INFO)
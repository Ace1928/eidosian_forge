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
def _get_brz_log_filename():
    """Return the brz log filename.

    :return: A path to the log file
    :raise EnvironmentError: If the cache directory could not be created
    """
    brz_log = os.environ.get('BRZ_LOG')
    if brz_log:
        return brz_log
    return os.path.join(bedding.cache_dir(), 'brz.log')
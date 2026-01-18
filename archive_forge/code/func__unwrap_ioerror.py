import io
import functools
import logging
import time
import warnings
import smart_open.bytebuffer
import smart_open.concurrency
import smart_open.utils
from smart_open import constants
def _unwrap_ioerror(ioe):
    """Given an IOError from _get, return the 'Error' dictionary from boto."""
    try:
        return ioe.backend_error.response['Error']
    except (AttributeError, KeyError):
        return None
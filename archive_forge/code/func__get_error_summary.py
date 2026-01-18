import datetime
import debtcollector
import functools
import io
import itertools
import logging
import logging.config
import logging.handlers
import re
import socket
import sys
import traceback
from dateutil import tz
from oslo_context import context as context_utils
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
def _get_error_summary(record):
    """Return the error summary

    If there is no active exception, return the default.

    If the record is being logged below the warning level, return an
    empty string.

    If there is an active exception, format it and return the
    resulting string.

    """
    error_summary = ''
    if record.levelno < logging.WARNING:
        return ''
    if record.exc_info:
        exc_info = record.exc_info
    else:
        exc_info = sys.exc_info()
        if not exc_info[0]:
            exc_info = None
        elif exc_info[0] in (TypeError, ValueError, KeyError, AttributeError, ImportError):
            exc_info = None
    if exc_info:
        try:
            error_summary = traceback.format_exception_only(exc_info[0], exc_info[1])[0].rstrip()
            if not record.exc_info:
                error_summary = error_summary.split('\n', 1)[0]
        except TypeError as type_err:
            error_summary = '<exception with %s>' % str(type_err)
        finally:
            del exc_info
    return error_summary
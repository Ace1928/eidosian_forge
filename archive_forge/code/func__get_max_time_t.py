import datetime
from functools import partial
import logging; log = logging.getLogger(__name__)
import sys
import time as _time
from passlib import exc
from passlib.utils.compat import unicode, u
from passlib.tests.utils import TestCase, time_call
from passlib import totp as totp_module
from passlib.totp import TOTP, AppWallet, AES_SUPPORT
def _get_max_time_t():
    """
    helper to calc max_time_t constant (see below)
    """
    value = 1 << 30
    year = 0
    while True:
        next_value = value << 1
        try:
            next_year = datetime.datetime.utcfromtimestamp(next_value - 1).year
        except (ValueError, OSError, OverflowError):
            break
        if next_year < year:
            break
        value = next_value
    value -= 1
    max_datetime_timestamp = 253402318800
    return min(value, max_datetime_timestamp)
from datetime import (
import time
import unittest
def _timezone(utc_offset):
    """
    Return a string representing the timezone offset.

    >>> _timezone(0)
    '+00:00'
    >>> _timezone(3600)
    '+01:00'
    >>> _timezone(-28800)
    '-08:00'
    >>> _timezone(-8 * 60 * 60)
    '-08:00'
    >>> _timezone(-30 * 60)
    '-00:30'
    """
    hours = abs(utc_offset) // 3600
    minutes = abs(utc_offset) % 3600 // 60
    sign = utc_offset < 0 and '-' or '+'
    return '%c%02d:%02d' % (sign, hours, minutes)
import sys
from . import case
from . import util
def _isnotsuite(test):
    """A crude way to tell apart testcases and suites with duck-typing"""
    try:
        iter(test)
    except TypeError:
        return True
    return False
from Markus Kuhn's C code, retrieved from:
from __future__ import division
import os
import sys
import warnings
from .table_wide import WIDE_EASTASIAN
from .table_zero import ZERO_WIDTH
from .unicode_versions import list_versions
@lru_cache(maxsize=128)
def _wcversion_value(ver_string):
    """
    Integer-mapped value of given dotted version string.

    :param str ver_string: Unicode version string, of form ``n.n.n``.
    :rtype: tuple(int)
    :returns: tuple of digit tuples, ``tuple(int, [...])``.
    """
    retval = tuple(map(int, ver_string.split('.')))
    return retval
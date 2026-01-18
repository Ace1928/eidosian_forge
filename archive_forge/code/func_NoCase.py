from __future__ import absolute_import
import types
from . import Errors
def NoCase(re):
    """
    NoCase(re) is an RE which matches the same strings as RE, but treating
    upper and lower case letters as equivalent.
    """
    return SwitchCase(re, nocase=1)
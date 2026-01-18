from __future__ import absolute_import
import types
from . import Errors
def Case(re):
    """
    Case(re) is an RE which matches the same strings as RE, but treating
    upper and lower case letters as distinct, i.e. it cancels the effect
    of any enclosing NoCase().
    """
    return SwitchCase(re, nocase=0)
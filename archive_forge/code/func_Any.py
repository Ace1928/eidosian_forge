from __future__ import absolute_import
import types
from . import Errors
def Any(s):
    """
    Any(s) is an RE which matches any character in the string |s|.
    """
    result = CodeRanges(chars_to_ranges(s))
    result.str = 'Any(%s)' % repr(s)
    return result
from __future__ import absolute_import
import types
from . import Errors
def AnyBut(s):
    """
    AnyBut(s) is an RE which matches any character (including
    newline) which is not in the string |s|.
    """
    ranges = chars_to_ranges(s)
    ranges.insert(0, -maxint)
    ranges.append(maxint)
    result = CodeRanges(ranges)
    result.str = 'AnyBut(%s)' % repr(s)
    return result
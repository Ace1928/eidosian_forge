from __future__ import absolute_import
import types
from . import Errors
def Range(s1, s2=None):
    """
    Range(c1, c2) is an RE which matches any single character in the range
    |c1| to |c2| inclusive.
    Range(s) where |s| is a string of even length is an RE which matches
    any single character in the ranges |s[0]| to |s[1]|, |s[2]| to |s[3]|,...
    """
    if s2:
        result = CodeRange(ord(s1), ord(s2) + 1)
        result.str = 'Range(%s,%s)' % (s1, s2)
    else:
        ranges = []
        for i in range(0, len(s1), 2):
            ranges.append(CodeRange(ord(s1[i]), ord(s1[i + 1]) + 1))
        result = Alt(*ranges)
        result.str = 'Range(%s)' % repr(s1)
    return result
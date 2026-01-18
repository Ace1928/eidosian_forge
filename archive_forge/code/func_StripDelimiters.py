from __future__ import print_function
import os
import re
import sys
def StripDelimiters(s, beg, end):
    if s[0] == beg:
        assert s[-1] == end
        return (s[1:-1], True)
    else:
        return (s, False)
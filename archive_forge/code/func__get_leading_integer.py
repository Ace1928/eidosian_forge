from collections import defaultdict
import re
from .pyutil import memoize
from .periodic import symbols
def _get_leading_integer(s):
    m = re.findall('^\\d+', s)
    if len(m) == 0:
        m = 1
    elif len(m) == 1:
        s = s[len(m[0]):]
        m = int(m[0])
    else:
        raise ValueError('Failed to parse: %s' % s)
    return (m, s)
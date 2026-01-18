import re
import operator
from fractions import Fraction
import sys
def _parse_variable(s):
    r = re.match('([_A-Za-z][_A-Za-z0-9]*)(.*)$', s)
    if r:
        return r.groups()
    else:
        return (None, s)
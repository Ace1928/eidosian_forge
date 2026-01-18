from collections import defaultdict
import re
from .pyutil import memoize
from .periodic import symbols
def _get_charge(chgstr):
    if chgstr == '+':
        return 1
    elif chgstr == '-':
        return -1
    for token, anti, sign in zip('+-', '-+', (1, -1)):
        if token in chgstr:
            if anti in chgstr:
                raise ValueError('Invalid charge description (+ & - present)')
            before, after = chgstr.split(token)
            if len(before) > 0 and len(after) > 0:
                raise ValueError('Values both before and after charge token')
            if len(after) > 0:
                return sign * int(1 if after == '' else after)
    raise ValueError('Invalid charge description (+ or - missing)')
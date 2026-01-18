import calendar
import datetime
import heapq
import itertools
import re
import sys
from functools import wraps
from warnings import warn
from six import advance_iterator, integer_types
from six.moves import _thread, range
from ._common import weekday as weekdaybase
def _handle_BYWEEKDAY(self, rrkwargs, name, value, **kwargs):
    """
        Two ways to specify this: +1MO or MO(+1)
        """
    l = []
    for wday in value.split(','):
        if '(' in wday:
            splt = wday.split('(')
            w = splt[0]
            n = int(splt[1][:-1])
        elif len(wday):
            for i in range(len(wday)):
                if wday[i] not in '+-0123456789':
                    break
            n = wday[:i] or None
            w = wday[i:]
            if n:
                n = int(n)
        else:
            raise ValueError('Invalid (empty) BYDAY specification.')
        l.append(weekdays[self._weekday_map[w]](n))
    rrkwargs['byweekday'] = l
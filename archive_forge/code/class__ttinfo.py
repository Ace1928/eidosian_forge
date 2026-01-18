import bisect
import calendar
import collections
import functools
import re
import weakref
from datetime import datetime, timedelta, tzinfo
from . import _common, _tzpath
class _ttinfo:
    __slots__ = ['utcoff', 'dstoff', 'tzname']

    def __init__(self, utcoff, dstoff, tzname):
        self.utcoff = utcoff
        self.dstoff = dstoff
        self.tzname = tzname

    def __eq__(self, other):
        return self.utcoff == other.utcoff and self.dstoff == other.dstoff and (self.tzname == other.tzname)

    def __repr__(self):
        return f'{self.__class__.__name__}' + f'({self.utcoff}, {self.dstoff}, {self.tzname})'
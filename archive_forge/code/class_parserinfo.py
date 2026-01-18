from __future__ import unicode_literals
import datetime
import re
import string
import time
import warnings
from calendar import monthrange
from io import StringIO
import six
from six import integer_types, text_type
from decimal import Decimal
from warnings import warn
from .. import relativedelta
from .. import tz
class parserinfo(object):
    """
    Class which handles what inputs are accepted. Subclass this to customize
    the language and acceptable values for each parameter.

    :param dayfirst:
        Whether to interpret the first value in an ambiguous 3-integer date
        (e.g. 01/05/09) as the day (``True``) or month (``False``). If
        ``yearfirst`` is set to ``True``, this distinguishes between YDM
        and YMD. Default is ``False``.

    :param yearfirst:
        Whether to interpret the first value in an ambiguous 3-integer date
        (e.g. 01/05/09) as the year. If ``True``, the first number is taken
        to be the year, otherwise the last number is taken to be the year.
        Default is ``False``.
    """
    JUMP = [' ', '.', ',', ';', '-', '/', "'", 'at', 'on', 'and', 'ad', 'm', 't', 'of', 'st', 'nd', 'rd', 'th']
    WEEKDAYS = [('Mon', 'Monday'), ('Tue', 'Tuesday'), ('Wed', 'Wednesday'), ('Thu', 'Thursday'), ('Fri', 'Friday'), ('Sat', 'Saturday'), ('Sun', 'Sunday')]
    MONTHS = [('Jan', 'January'), ('Feb', 'February'), ('Mar', 'March'), ('Apr', 'April'), ('May', 'May'), ('Jun', 'June'), ('Jul', 'July'), ('Aug', 'August'), ('Sep', 'Sept', 'September'), ('Oct', 'October'), ('Nov', 'November'), ('Dec', 'December')]
    HMS = [('h', 'hour', 'hours'), ('m', 'minute', 'minutes'), ('s', 'second', 'seconds')]
    AMPM = [('am', 'a'), ('pm', 'p')]
    UTCZONE = ['UTC', 'GMT', 'Z', 'z']
    PERTAIN = ['of']
    TZOFFSET = {}

    def __init__(self, dayfirst=False, yearfirst=False):
        self._jump = self._convert(self.JUMP)
        self._weekdays = self._convert(self.WEEKDAYS)
        self._months = self._convert(self.MONTHS)
        self._hms = self._convert(self.HMS)
        self._ampm = self._convert(self.AMPM)
        self._utczone = self._convert(self.UTCZONE)
        self._pertain = self._convert(self.PERTAIN)
        self.dayfirst = dayfirst
        self.yearfirst = yearfirst
        self._year = time.localtime().tm_year
        self._century = self._year // 100 * 100

    def _convert(self, lst):
        dct = {}
        for i, v in enumerate(lst):
            if isinstance(v, tuple):
                for v in v:
                    dct[v.lower()] = i
            else:
                dct[v.lower()] = i
        return dct

    def jump(self, name):
        return name.lower() in self._jump

    def weekday(self, name):
        try:
            return self._weekdays[name.lower()]
        except KeyError:
            pass
        return None

    def month(self, name):
        try:
            return self._months[name.lower()] + 1
        except KeyError:
            pass
        return None

    def hms(self, name):
        try:
            return self._hms[name.lower()]
        except KeyError:
            return None

    def ampm(self, name):
        try:
            return self._ampm[name.lower()]
        except KeyError:
            return None

    def pertain(self, name):
        return name.lower() in self._pertain

    def utczone(self, name):
        return name.lower() in self._utczone

    def tzoffset(self, name):
        if name in self._utczone:
            return 0
        return self.TZOFFSET.get(name)

    def convertyear(self, year, century_specified=False):
        """
        Converts two-digit years to year within [-50, 49]
        range of self._year (current local time)
        """
        assert year >= 0
        if year < 100 and (not century_specified):
            year += self._century
            if year >= self._year + 50:
                year -= 100
            elif year < self._year - 50:
                year += 100
        return year

    def validate(self, res):
        if res.year is not None:
            res.year = self.convertyear(res.year, res.century_specified)
        if res.tzoffset == 0 and (not res.tzname) or (res.tzname == 'Z' or res.tzname == 'z'):
            res.tzname = 'UTC'
            res.tzoffset = 0
        elif res.tzoffset != 0 and res.tzname and self.utczone(res.tzname):
            res.tzoffset = 0
        return True
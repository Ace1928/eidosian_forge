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
def _build_tzaware(self, naive, res, tzinfos):
    if callable(tzinfos) or (tzinfos and res.tzname in tzinfos):
        tzinfo = self._build_tzinfo(tzinfos, res.tzname, res.tzoffset)
        aware = naive.replace(tzinfo=tzinfo)
        aware = self._assign_tzname(aware, res.tzname)
    elif res.tzname and res.tzname in time.tzname:
        aware = naive.replace(tzinfo=tz.tzlocal())
        aware = self._assign_tzname(aware, res.tzname)
        if aware.tzname() != res.tzname and res.tzname in self.info.UTCZONE:
            aware = aware.replace(tzinfo=tz.UTC)
    elif res.tzoffset == 0:
        aware = naive.replace(tzinfo=tz.UTC)
    elif res.tzoffset:
        aware = naive.replace(tzinfo=tz.tzoffset(res.tzname, res.tzoffset))
    elif not res.tzname and (not res.tzoffset):
        aware = naive
    elif res.tzname:
        warnings.warn('tzname {tzname} identified but not understood.  Pass `tzinfos` argument in order to correctly return a timezone-aware datetime.  In a future version, this will raise an exception.'.format(tzname=res.tzname), category=UnknownTimezoneWarning)
        aware = naive
    return aware
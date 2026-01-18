import datetime
import functools
import logging
import re
from dateutil.rrule import (rrule, MO, TU, WE, TH, FR, SA, SU, YEARLY,
from dateutil.relativedelta import relativedelta
import dateutil.parser
import dateutil.tz
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, ticker, units
def _update_rrule(self, **kwargs):
    tzinfo = self._base_tzinfo
    if 'dtstart' in kwargs:
        dtstart = kwargs['dtstart']
        if dtstart.tzinfo is not None:
            if tzinfo is None:
                tzinfo = dtstart.tzinfo
            else:
                dtstart = dtstart.astimezone(tzinfo)
            kwargs['dtstart'] = dtstart.replace(tzinfo=None)
    if 'until' in kwargs:
        until = kwargs['until']
        if until.tzinfo is not None:
            if tzinfo is not None:
                until = until.astimezone(tzinfo)
            else:
                raise ValueError('until cannot be aware if dtstart is naive and tzinfo is None')
            kwargs['until'] = until.replace(tzinfo=None)
    self._construct = kwargs.copy()
    self._tzinfo = tzinfo
    self._rrule = rrule(**self._construct)
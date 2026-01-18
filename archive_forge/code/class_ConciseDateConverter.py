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
class ConciseDateConverter(DateConverter):

    def __init__(self, formats=None, zero_formats=None, offset_formats=None, show_offset=True, *, interval_multiples=True):
        self._formats = formats
        self._zero_formats = zero_formats
        self._offset_formats = offset_formats
        self._show_offset = show_offset
        self._interval_multiples = interval_multiples
        super().__init__()

    def axisinfo(self, unit, axis):
        tz = unit
        majloc = AutoDateLocator(tz=tz, interval_multiples=self._interval_multiples)
        majfmt = ConciseDateFormatter(majloc, tz=tz, formats=self._formats, zero_formats=self._zero_formats, offset_formats=self._offset_formats, show_offset=self._show_offset)
        datemin = datetime.date(1970, 1, 1)
        datemax = datetime.date(1970, 1, 2)
        return units.AxisInfo(majloc=majloc, majfmt=majfmt, label='', default_limits=(datemin, datemax))
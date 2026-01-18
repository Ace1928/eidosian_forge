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
def _create_rrule(self, vmin, vmax):
    ymin = max(self.base.le(vmin.year) * self.base.step, 1)
    ymax = min(self.base.ge(vmax.year) * self.base.step, 9999)
    c = self.rule._construct
    replace = {'year': ymin, 'month': c.get('bymonth', 1), 'day': c.get('bymonthday', 1), 'hour': 0, 'minute': 0, 'second': 0}
    start = vmin.replace(**replace)
    stop = start.replace(year=ymax)
    self.rule.set(dtstart=start, until=stop)
    return (start, stop)
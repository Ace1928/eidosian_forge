import datetime
import calendar
import operator
from math import copysign
from six import integer_types
from warnings import warn
from ._common import weekday
def _set_months(self, months):
    self.months = months
    if abs(self.months) > 11:
        s = _sign(self.months)
        div, mod = divmod(self.months * s, 12)
        self.months = mod * s
        self.years = div * s
    else:
        self.years = 0
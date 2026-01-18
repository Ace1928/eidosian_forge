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
def _assign_tzname(self, dt, tzname):
    if dt.tzname() != tzname:
        new_dt = tz.enfold(dt, fold=1)
        if new_dt.tzname() == tzname:
            return new_dt
    return dt
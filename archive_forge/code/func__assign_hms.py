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
def _assign_hms(self, res, value_repr, hms):
    value = self._to_decimal(value_repr)
    if hms == 0:
        res.hour = int(value)
        if value % 1:
            res.minute = int(60 * (value % 1))
    elif hms == 1:
        res.minute, res.second = self._parse_min_sec(value)
    elif hms == 2:
        res.second, res.microsecond = self._parsems(value_repr)
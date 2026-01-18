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
def _ampm_valid(self, hour, ampm, fuzzy):
    """
        For fuzzy parsing, 'a' or 'am' (both valid English words)
        may erroneously trigger the AM/PM flag. Deal with that
        here.
        """
    val_is_ampm = True
    if fuzzy and ampm is not None:
        val_is_ampm = False
    if hour is None:
        if fuzzy:
            val_is_ampm = False
        else:
            raise ValueError('No hour specified with AM or PM flag.')
    elif not 0 <= hour <= 12:
        if fuzzy:
            val_is_ampm = False
        else:
            raise ValueError('Invalid hour specified for 12-hour clock.')
    return val_is_ampm
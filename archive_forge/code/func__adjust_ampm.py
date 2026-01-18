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
def _adjust_ampm(self, hour, ampm):
    if hour < 12 and ampm == 1:
        hour += 12
    elif hour == 12 and ampm == 0:
        hour = 0
    return hour
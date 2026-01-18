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
def _to_decimal(self, val):
    try:
        decimal_value = Decimal(val)
        if not decimal_value.is_finite():
            raise ValueError('Converted decimal value is infinite or NaN')
    except Exception as e:
        msg = 'Could not convert %s to decimal' % val
        six.raise_from(ValueError(msg), e)
    else:
        return decimal_value
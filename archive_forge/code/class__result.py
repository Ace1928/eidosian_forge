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
class _result(_resultbase):
    __slots__ = ['stdabbr', 'stdoffset', 'dstabbr', 'dstoffset', 'start', 'end']

    class _attr(_resultbase):
        __slots__ = ['month', 'week', 'weekday', 'yday', 'jyday', 'day', 'time']

    def __repr__(self):
        return self._repr('')

    def __init__(self):
        _resultbase.__init__(self)
        self.start = self._attr()
        self.end = self._attr()
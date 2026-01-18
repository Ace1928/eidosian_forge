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
class _resultbase(object):

    def __init__(self):
        for attr in self.__slots__:
            setattr(self, attr, None)

    def _repr(self, classname):
        l = []
        for attr in self.__slots__:
            value = getattr(self, attr)
            if value is not None:
                l.append('%s=%s' % (attr, repr(value)))
        return '%s(%s)' % (classname, ', '.join(l))

    def __len__(self):
        return sum((getattr(self, attr) is not None for attr in self.__slots__))

    def __repr__(self):
        return self._repr(self.__class__.__name__)
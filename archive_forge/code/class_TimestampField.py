from bisect import bisect_left
from bisect import bisect_right
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from inspect import isclass
import calendar
import collections
import datetime
import decimal
import hashlib
import itertools
import logging
import operator
import re
import socket
import struct
import sys
import threading
import time
import uuid
import warnings
class TimestampField(BigIntegerField):
    valid_resolutions = [10 ** i for i in range(7)]

    def __init__(self, *args, **kwargs):
        self.resolution = kwargs.pop('resolution', None)
        if not self.resolution:
            self.resolution = 1
        elif self.resolution in range(2, 7):
            self.resolution = 10 ** self.resolution
        elif self.resolution not in self.valid_resolutions:
            raise ValueError('TimestampField resolution must be one of: %s' % ', '.join((str(i) for i in self.valid_resolutions)))
        self.ticks_to_microsecond = 1000000 // self.resolution
        self.utc = kwargs.pop('utc', False) or False
        dflt = utcnow if self.utc else datetime.datetime.now
        kwargs.setdefault('default', dflt)
        super(TimestampField, self).__init__(*args, **kwargs)

    def local_to_utc(self, dt):
        return datetime.datetime(*time.gmtime(time.mktime(dt.timetuple()))[:6])

    def utc_to_local(self, dt):
        ts = calendar.timegm(dt.utctimetuple())
        return datetime.datetime.fromtimestamp(ts)

    def get_timestamp(self, value):
        if self.utc:
            return calendar.timegm(value.utctimetuple())
        else:
            return time.mktime(value.timetuple())

    def db_value(self, value):
        if value is None:
            return
        if isinstance(value, datetime.datetime):
            pass
        elif isinstance(value, datetime.date):
            value = datetime.datetime(value.year, value.month, value.day)
        else:
            return int(round(value * self.resolution))
        timestamp = self.get_timestamp(value)
        if self.resolution > 1:
            timestamp += value.microsecond * 1e-06
            timestamp *= self.resolution
        return int(round(timestamp))

    def python_value(self, value):
        if value is not None and isinstance(value, (int, float, long)):
            if self.resolution > 1:
                value, ticks = divmod(value, self.resolution)
                microseconds = int(ticks * self.ticks_to_microsecond)
            else:
                microseconds = 0
            if self.utc:
                value = utcfromtimestamp(value)
            else:
                value = datetime.datetime.fromtimestamp(value)
            if microseconds:
                value = value.replace(microsecond=microseconds)
        return value

    def from_timestamp(self):
        expr = self / Value(self.resolution, converter=False) if self.resolution > 1 else self
        return self.model._meta.database.from_timestamp(expr)
    year = property(_timestamp_date_part('year'))
    month = property(_timestamp_date_part('month'))
    day = property(_timestamp_date_part('day'))
    hour = property(_timestamp_date_part('hour'))
    minute = property(_timestamp_date_part('minute'))
    second = property(_timestamp_date_part('second'))
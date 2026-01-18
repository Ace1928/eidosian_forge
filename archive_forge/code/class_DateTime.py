import base64
import sys
import time
from datetime import datetime
from decimal import Decimal
import http.client
import urllib.parse
from xml.parsers import expat
import errno
from io import BytesIO
class DateTime:
    """DateTime wrapper for an ISO 8601 string or time tuple or
    localtime integer value to generate 'dateTime.iso8601' XML-RPC
    value.
    """

    def __init__(self, value=0):
        if isinstance(value, str):
            self.value = value
        else:
            self.value = _strftime(value)

    def make_comparable(self, other):
        if isinstance(other, DateTime):
            s = self.value
            o = other.value
        elif isinstance(other, datetime):
            s = self.value
            o = _iso8601_format(other)
        elif isinstance(other, str):
            s = self.value
            o = other
        elif hasattr(other, 'timetuple'):
            s = self.timetuple()
            o = other.timetuple()
        else:
            s = self
            o = NotImplemented
        return (s, o)

    def __lt__(self, other):
        s, o = self.make_comparable(other)
        if o is NotImplemented:
            return NotImplemented
        return s < o

    def __le__(self, other):
        s, o = self.make_comparable(other)
        if o is NotImplemented:
            return NotImplemented
        return s <= o

    def __gt__(self, other):
        s, o = self.make_comparable(other)
        if o is NotImplemented:
            return NotImplemented
        return s > o

    def __ge__(self, other):
        s, o = self.make_comparable(other)
        if o is NotImplemented:
            return NotImplemented
        return s >= o

    def __eq__(self, other):
        s, o = self.make_comparable(other)
        if o is NotImplemented:
            return NotImplemented
        return s == o

    def timetuple(self):
        return time.strptime(self.value, '%Y%m%dT%H:%M:%S')

    def __str__(self):
        return self.value

    def __repr__(self):
        return '<%s %r at %#x>' % (self.__class__.__name__, self.value, id(self))

    def decode(self, data):
        self.value = str(data).strip()

    def encode(self, out):
        out.write('<value><dateTime.iso8601>')
        out.write(self.value)
        out.write('</dateTime.iso8601></value>\n')
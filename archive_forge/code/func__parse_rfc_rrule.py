import calendar
import datetime
import heapq
import itertools
import re
import sys
from functools import wraps
from warnings import warn
from six import advance_iterator, integer_types
from six.moves import _thread, range
from ._common import weekday as weekdaybase
def _parse_rfc_rrule(self, line, dtstart=None, cache=False, ignoretz=False, tzinfos=None):
    if line.find(':') != -1:
        name, value = line.split(':')
        if name != 'RRULE':
            raise ValueError('unknown parameter name')
    else:
        value = line
    rrkwargs = {}
    for pair in value.split(';'):
        name, value = pair.split('=')
        name = name.upper()
        value = value.upper()
        try:
            getattr(self, '_handle_' + name)(rrkwargs, name, value, ignoretz=ignoretz, tzinfos=tzinfos)
        except AttributeError:
            raise ValueError("unknown parameter '%s'" % name)
        except (KeyError, ValueError):
            raise ValueError("invalid '%s': %s" % (name, value))
    return rrule(dtstart=dtstart, cache=cache, **rrkwargs)
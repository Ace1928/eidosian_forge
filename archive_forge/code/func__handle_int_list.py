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
def _handle_int_list(self, rrkwargs, name, value, **kwargs):
    rrkwargs[name.lower()] = [int(x) for x in value.split(',')]
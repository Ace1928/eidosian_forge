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
class weekday(weekdaybase):
    """
    This version of weekday does not allow n = 0.
    """

    def __init__(self, wkday, n=None):
        if n == 0:
            raise ValueError("Can't create weekday with n==0")
        super(weekday, self).__init__(wkday, n)
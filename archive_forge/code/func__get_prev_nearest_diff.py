from __future__ import absolute_import, print_function, division
import traceback as _traceback
import copy
import math
import re
import sys
import inspect
from time import time
import datetime
from dateutil.relativedelta import relativedelta
from dateutil.tz import tzutc
import calendar
import binascii
import random
import pytz  # noqa
def _get_prev_nearest_diff(self, x, to_check, range_val):
    candidates = to_check[:]
    candidates.reverse()
    for d in candidates:
        if d != 'l' and d <= x:
            return d - x
    if 'l' in candidates:
        return -x
    candidate = candidates[0]
    for c in candidates:
        if c <= range_val:
            candidate = c
            break
    if candidate > range_val:
        return -x
    return candidate - x - range_val
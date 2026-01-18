import calendar
import datetime
import functools
import logging
import time
import iso8601
from oslo_utils import reflection
@staticmethod
def _delta_seconds(earlier, later):
    return max(0.0, later - earlier)
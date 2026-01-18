import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def _get_next_mw_duration(self):
    """
        Returns the duration of the next maintenance window.

        :return:  Duration of next maintenance window (or None if no
                  maintenance windows are scheduled)
        :rtype:   :class:`datetime.timedelta` or ``None``
        """
    next_window = self._get_next_maint()
    if not next_window:
        return None
    next_begin = timestamp_to_datetime(next_window['beginTime'])
    next_end = timestamp_to_datetime(next_window['endTime'])
    return next_end - next_begin
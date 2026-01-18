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
class GCEZone(NodeLocation):
    """Subclass of NodeLocation to provide additional information."""

    def __init__(self, id, name, status, maintenance_windows, deprecated, driver, extra=None):
        self.status = status
        self.maintenance_windows = maintenance_windows
        self.deprecated = deprecated
        self.extra = extra
        country = name.split('-')[0]
        super().__init__(id=str(id), name=name, country=country, driver=driver, extra=extra)

    @property
    def time_until_mw(self):
        """
        Returns the time until the next Maintenance Window as a
        datetime.timedelta object.
        """
        return self._get_time_until_mw()

    @property
    def next_mw_duration(self):
        """
        Returns the duration of the next Maintenance Window as a
        datetime.timedelta object.
        """
        return self._get_next_mw_duration()

    def _now(self):
        """
        Returns current UTC time.

        Can be overridden in unittests.
        """
        return datetime.datetime.utcnow()

    def _get_next_maint(self):
        """
        Returns the next Maintenance Window.

        :return:  A dictionary containing maintenance window info (or None if
                  no maintenance windows are scheduled)
                  The dictionary contains 4 keys with values of type ``str``
                      - name: The name of the maintenance window
                      - description: Description of the maintenance window
                      - beginTime: RFC3339 Timestamp
                      - endTime: RFC3339 Timestamp
        :rtype:   ``dict`` or ``None``
        """
        begin = None
        next_window = None
        if not self.maintenance_windows:
            return None
        if len(self.maintenance_windows) == 1:
            return self.maintenance_windows[0]
        for mw in self.maintenance_windows:
            begin_next = timestamp_to_datetime(mw['beginTime'])
            if not begin or begin_next < begin:
                begin = begin_next
                next_window = mw
        return next_window

    def _get_time_until_mw(self):
        """
        Returns time until next maintenance window.

        :return:  Time until next maintenance window (or None if no
                  maintenance windows are scheduled)
        :rtype:   :class:`datetime.timedelta` or ``None``
        """
        next_window = self._get_next_maint()
        if not next_window:
            return None
        now = self._now()
        next_begin = timestamp_to_datetime(next_window['beginTime'])
        return next_begin - now

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

    def __repr__(self):
        return '<GCEZone id="{}" name="{}" status="{}">'.format(self.id, self.name, self.status)
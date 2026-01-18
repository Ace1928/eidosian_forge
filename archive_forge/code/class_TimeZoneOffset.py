from __future__ import with_statement
import datetime
import functools
import inspect
import logging
import os
import re
import sys
import six
class TimeZoneOffset(datetime.tzinfo):
    """Time zone information as encoded/decoded for DateTimeFields."""

    def __init__(self, offset):
        """Initialize a time zone offset.

        Args:
          offset: Integer or timedelta time zone offset, in minutes from UTC.
            This can be negative.
        """
        super(TimeZoneOffset, self).__init__()
        if isinstance(offset, datetime.timedelta):
            offset = total_seconds(offset) / 60
        self.__offset = offset

    def utcoffset(self, _):
        """Get the a timedelta with the time zone's offset from UTC.

        Returns:
          The time zone offset from UTC, as a timedelta.
        """
        return datetime.timedelta(minutes=self.__offset)

    def dst(self, _):
        """Get the daylight savings time offset.

        The formats that ProtoRPC uses to encode/decode time zone
        information don't contain any information about daylight
        savings time. So this always returns a timedelta of 0.

        Returns:
          A timedelta of 0.

        """
        return datetime.timedelta(0)
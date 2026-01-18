from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import datetime
import io
import re
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import times
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import urllib
def TransformDate(r, format='%Y-%m-%dT%H:%M:%S', unit=1, undefined='', tz=None, tz_default=None):
    """Formats the resource as a strftime() format.

  Args:
    r: A timestamp number or an object with 3 or more of these fields: year,
      month, day, hour, minute, second, millisecond, microsecond, nanosecond.
    format: The strftime(3) format.
    unit: If the resource is a Timestamp then divide by _unit_ to yield seconds.
    undefined: Returns this value if the resource is not a valid time.
    tz: Return the time relative to the tz timezone if specified, the explicit
      timezone in the resource if it has one, otherwise the local timezone.
      For example: `date(tz=EST5EDT, tz_default=UTC)`.
    tz_default: The default timezone if the resource does not have a timezone
      suffix.

  Returns:
    The strftime() date format for r or undefined if r does not contain a valid
    time.
  """
    try:
        r = r.isoformat()
    except (AttributeError, TypeError, ValueError):
        pass
    tz_in = times.GetTimeZone(tz_default) if tz_default else None
    try:
        timestamp = float(r) / float(unit)
    except (TypeError, ValueError):
        timestamp = None
    if timestamp is not None:
        try:
            dt = times.GetDateTimeFromTimeStamp(timestamp, tz_in)
            return times.FormatDateTime(dt, format)
        except times.Error:
            pass
    original_repr = resource_property.Get(r, ['datetime'], None)
    if original_repr and isinstance(original_repr, six.string_types):
        r = original_repr
    tz_out = times.GetTimeZone(tz) if tz else None
    try:
        dt = times.ParseDateTime(r, tzinfo=tz_in)
        return times.FormatDateTime(dt, format, tz_out)
    except times.Error:
        pass

    def _FormatFromParts():
        """Returns the formatted time from broken down time parts in r.

    Raises:
      TypeError: For invalid time part errors.
      ValueError: For time conversion errors or not enough valid time parts.

    Returns:
      The formatted time from broken down time parts in r.
    """
        valid = 0
        parts = []
        now = datetime.datetime.now(tz_in)
        for part in ('year', 'month', 'day', 'hour', 'minute', 'second'):
            value = resource_property.Get(r, [part], None)
            if value is None:
                value = getattr(now, part, 0)
            else:
                valid += 1
            parts.append(int(value))
        parts.append(0)
        for i, part in enumerate(['nanosecond', 'microsecond', 'millisecond']):
            value = resource_property.Get(r, [part], None)
            if value is not None:
                parts[-1] += int(int(value) * 1000 ** (i - 1))
        if valid < 3:
            raise ValueError
        parts.append(tz_in)
        dt = datetime.datetime(*parts)
        return times.FormatDateTime(dt, format, tz_out)
    try:
        return _FormatFromParts()
    except (TypeError, ValueError):
        pass
    return undefined
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
def TransformDuration(r, start='', end='', parts=3, precision=3, calendar=True, unit=1, undefined=''):
    """Formats the resource as an ISO 8601 duration string.

  The [ISO 8601 Duration](https://en.wikipedia.org/wiki/ISO_8601#Durations)
  format is: "[-]P[nY][nM][nD][T[nH][nM][n[.m]S]]". The 0 duration is "P0".
  Otherwise at least one part will always be displayed. Negative durations are
  prefixed by "-". "T" disambiguates months "P2M" to the left of "T" and minutes
  "PT5M" to the right.

  If the resource is a datetime then the duration of `resource - current_time`
  is returned.

  Args:
    r: A JSON-serializable object.
    start: The name of a start time attribute in the resource. The duration of
      the `end - start` time attributes in resource is returned. If `end` is
      not specified then the current time is used.
    end: The name of an end time attribute in the resource. Defaults to
      the current time if omitted. Ignored if `start` is not specified.
    parts: Format at most this many duration parts starting with largest
      non-zero part.
    precision: Format the last duration part with precision digits after the
      decimal point. Trailing "0" and "." are always stripped.
    calendar: Allow time units larger than hours in formatted durations if true.
      Durations specifying hours or smaller units are exact across daylight
      savings time boundaries. On by default. Use calendar=false to disable.
      For example, if `calendar=true` then at the daylight savings boundary
      2016-03-13T01:00:00 + P1D => 2016-03-14T01:00:00 but 2016-03-13T01:00:00 +
      PT24H => 2016-03-14T03:00:00. Similarly, a +P1Y duration will be inexact
      but "calendar correct", yielding the same month and day number next year,
      even in leap years.
    unit: Divide the resource numeric value by _unit_ to yield seconds.
    undefined: Returns this value if the resource is not a valid timestamp.

  Returns:
    The ISO 8601 duration string for r or undefined if r is not a duration.

  Example:
    `duration(start=createTime,end=updateTime)`:::
    The duration from resource creation to the most recent update.
    `updateTime.duration()`:::
    The duration since the most recent resource update.
  """
    try:
        parts = int(parts)
        precision = int(precision)
    except ValueError:
        return undefined
    calendar = GetBooleanArgValue(calendar)
    if start:
        try:
            start_datetime = times.ParseDateTime(GetKeyValue(r, start))
            end_value = GetKeyValue(r, end) if end else None
            if end_value:
                end_datetime = times.ParseDateTime(end_value)
            else:
                end_datetime = times.Now(tzinfo=start_datetime.tzinfo)
        except times.Error:
            return undefined
        delta = end_datetime - start_datetime
        return times.GetDurationFromTimeDelta(delta=delta, calendar=calendar).Format(parts=parts, precision=precision)
    try:
        seconds = float(r) / float(unit)
    except (TypeError, ValueError):
        seconds = None
    if seconds is not None:
        try:
            duration = times.ParseDuration('PT{0}S'.format(seconds), calendar=calendar)
            return duration.Format(parts=parts, precision=precision)
        except times.Error:
            pass
    try:
        duration = times.ParseDuration(r)
        return duration.Format(parts=parts, precision=precision)
    except times.Error:
        pass
    try:
        start_datetime = times.ParseDateTime(r)
    except times.Error:
        return undefined
    end_datetime = times.Now(tzinfo=start_datetime.tzinfo)
    delta = end_datetime - start_datetime
    return times.GetDurationFromTimeDelta(delta=delta, calendar=calendar).Format(parts=parts, precision=precision)
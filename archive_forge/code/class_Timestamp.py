import calendar
import collections.abc
import datetime
import warnings
from google.protobuf.internal import field_mask
class Timestamp(object):
    """Class for Timestamp message type."""
    __slots__ = ()

    def ToJsonString(self):
        """Converts Timestamp to RFC 3339 date string format.

    Returns:
      A string converted from timestamp. The string is always Z-normalized
      and uses 3, 6 or 9 fractional digits as required to represent the
      exact time. Example of the return format: '1972-01-01T10:00:20.021Z'
    """
        _CheckTimestampValid(self.seconds, self.nanos)
        nanos = self.nanos
        seconds = self.seconds % _SECONDS_PER_DAY
        days = (self.seconds - seconds) // _SECONDS_PER_DAY
        dt = datetime.datetime(1970, 1, 1) + datetime.timedelta(days, seconds)
        result = dt.isoformat()
        if nanos % 1000000000.0 == 0:
            return result + 'Z'
        if nanos % 1000000.0 == 0:
            return result + '.%03dZ' % (nanos / 1000000.0)
        if nanos % 1000.0 == 0:
            return result + '.%06dZ' % (nanos / 1000.0)
        return result + '.%09dZ' % nanos

    def FromJsonString(self, value):
        """Parse a RFC 3339 date string format to Timestamp.

    Args:
      value: A date string. Any fractional digits (or none) and any offset are
          accepted as long as they fit into nano-seconds precision.
          Example of accepted format: '1972-01-01T10:00:20.021-05:00'

    Raises:
      ValueError: On parsing problems.
    """
        if not isinstance(value, str):
            raise ValueError('Timestamp JSON value not a string: {!r}'.format(value))
        timezone_offset = value.find('Z')
        if timezone_offset == -1:
            timezone_offset = value.find('+')
        if timezone_offset == -1:
            timezone_offset = value.rfind('-')
        if timezone_offset == -1:
            raise ValueError('Failed to parse timestamp: missing valid timezone offset.')
        time_value = value[0:timezone_offset]
        point_position = time_value.find('.')
        if point_position == -1:
            second_value = time_value
            nano_value = ''
        else:
            second_value = time_value[:point_position]
            nano_value = time_value[point_position + 1:]
        if 't' in second_value:
            raise ValueError("time data '{0}' does not match format '%Y-%m-%dT%H:%M:%S', lowercase 't' is not accepted".format(second_value))
        date_object = datetime.datetime.strptime(second_value, _TIMESTAMPFOMAT)
        td = date_object - datetime.datetime(1970, 1, 1)
        seconds = td.seconds + td.days * _SECONDS_PER_DAY
        if len(nano_value) > 9:
            raise ValueError('Failed to parse Timestamp: nanos {0} more than 9 fractional digits.'.format(nano_value))
        if nano_value:
            nanos = round(float('0.' + nano_value) * 1000000000.0)
        else:
            nanos = 0
        if value[timezone_offset] == 'Z':
            if len(value) != timezone_offset + 1:
                raise ValueError('Failed to parse timestamp: invalid trailing data {0}.'.format(value))
        else:
            timezone = value[timezone_offset:]
            pos = timezone.find(':')
            if pos == -1:
                raise ValueError('Invalid timezone offset value: {0}.'.format(timezone))
            if timezone[0] == '+':
                seconds -= (int(timezone[1:pos]) * 60 + int(timezone[pos + 1:])) * 60
            else:
                seconds += (int(timezone[1:pos]) * 60 + int(timezone[pos + 1:])) * 60
        _CheckTimestampValid(seconds, nanos)
        self.seconds = int(seconds)
        self.nanos = int(nanos)

    def GetCurrentTime(self):
        """Get the current UTC into Timestamp."""
        self.FromDatetime(datetime.datetime.utcnow())

    def ToNanoseconds(self):
        """Converts Timestamp to nanoseconds since epoch."""
        _CheckTimestampValid(self.seconds, self.nanos)
        return self.seconds * _NANOS_PER_SECOND + self.nanos

    def ToMicroseconds(self):
        """Converts Timestamp to microseconds since epoch."""
        _CheckTimestampValid(self.seconds, self.nanos)
        return self.seconds * _MICROS_PER_SECOND + self.nanos // _NANOS_PER_MICROSECOND

    def ToMilliseconds(self):
        """Converts Timestamp to milliseconds since epoch."""
        _CheckTimestampValid(self.seconds, self.nanos)
        return self.seconds * _MILLIS_PER_SECOND + self.nanos // _NANOS_PER_MILLISECOND

    def ToSeconds(self):
        """Converts Timestamp to seconds since epoch."""
        _CheckTimestampValid(self.seconds, self.nanos)
        return self.seconds

    def FromNanoseconds(self, nanos):
        """Converts nanoseconds since epoch to Timestamp."""
        seconds = nanos // _NANOS_PER_SECOND
        nanos = nanos % _NANOS_PER_SECOND
        _CheckTimestampValid(seconds, nanos)
        self.seconds = seconds
        self.nanos = nanos

    def FromMicroseconds(self, micros):
        """Converts microseconds since epoch to Timestamp."""
        seconds = micros // _MICROS_PER_SECOND
        nanos = micros % _MICROS_PER_SECOND * _NANOS_PER_MICROSECOND
        _CheckTimestampValid(seconds, nanos)
        self.seconds = seconds
        self.nanos = nanos

    def FromMilliseconds(self, millis):
        """Converts milliseconds since epoch to Timestamp."""
        seconds = millis // _MILLIS_PER_SECOND
        nanos = millis % _MILLIS_PER_SECOND * _NANOS_PER_MILLISECOND
        _CheckTimestampValid(seconds, nanos)
        self.seconds = seconds
        self.nanos = nanos

    def FromSeconds(self, seconds):
        """Converts seconds since epoch to Timestamp."""
        _CheckTimestampValid(seconds, 0)
        self.seconds = seconds
        self.nanos = 0

    def ToDatetime(self, tzinfo=None):
        """Converts Timestamp to a datetime.

    Args:
      tzinfo: A datetime.tzinfo subclass; defaults to None.

    Returns:
      If tzinfo is None, returns a timezone-naive UTC datetime (with no timezone
      information, i.e. not aware that it's UTC).

      Otherwise, returns a timezone-aware datetime in the input timezone.
    """
        _CheckTimestampValid(self.seconds, self.nanos)
        delta = datetime.timedelta(seconds=self.seconds, microseconds=_RoundTowardZero(self.nanos, _NANOS_PER_MICROSECOND))
        if tzinfo is None:
            return _EPOCH_DATETIME_NAIVE + delta
        else:
            return (_EPOCH_DATETIME_AWARE + delta).astimezone(tzinfo)

    def FromDatetime(self, dt):
        """Converts datetime to Timestamp.

    Args:
      dt: A datetime. If it's timezone-naive, it's assumed to be in UTC.
    """
        seconds = calendar.timegm(dt.utctimetuple())
        nanos = dt.microsecond * _NANOS_PER_MICROSECOND
        _CheckTimestampValid(seconds, nanos)
        self.seconds = seconds
        self.nanos = nanos
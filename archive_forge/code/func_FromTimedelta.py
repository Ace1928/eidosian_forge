import calendar
import collections.abc
import datetime
import warnings
from google.protobuf.internal import field_mask
def FromTimedelta(self, td):
    """Converts timedelta to Duration."""
    self._NormalizeDuration(td.seconds + td.days * _SECONDS_PER_DAY, td.microseconds * _NANOS_PER_MICROSECOND)
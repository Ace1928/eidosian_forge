import calendar
import collections.abc
import datetime
import warnings
from google.protobuf.internal import field_mask
def FromMicroseconds(self, micros):
    """Converts microseconds to Duration."""
    self._NormalizeDuration(micros // _MICROS_PER_SECOND, micros % _MICROS_PER_SECOND * _NANOS_PER_MICROSECOND)
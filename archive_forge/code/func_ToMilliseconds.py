import calendar
import collections.abc
import datetime
import warnings
from google.protobuf.internal import field_mask
def ToMilliseconds(self):
    """Converts a Duration to milliseconds."""
    millis = _RoundTowardZero(self.nanos, _NANOS_PER_MILLISECOND)
    return self.seconds * _MILLIS_PER_SECOND + millis
import calendar
import collections.abc
import datetime
import warnings
from google.protobuf.internal import field_mask
def _CheckDurationValid(seconds, nanos):
    if seconds < -_DURATION_SECONDS_MAX or seconds > _DURATION_SECONDS_MAX:
        raise ValueError('Duration is not valid: Seconds {0} must be in range [-315576000000, 315576000000].'.format(seconds))
    if nanos <= -_NANOS_PER_SECOND or nanos >= _NANOS_PER_SECOND:
        raise ValueError('Duration is not valid: Nanos {0} must be in range [-999999999, 999999999].'.format(nanos))
    if nanos < 0 and seconds > 0 or (nanos > 0 and seconds < 0):
        raise ValueError('Duration is not valid: Sign mismatch.')
import calendar
import collections.abc
import datetime
import warnings
from google.protobuf.internal import field_mask
def _NormalizeDuration(self, seconds, nanos):
    """Set Duration by seconds and nanos."""
    if seconds < 0 and nanos > 0:
        seconds += 1
        nanos -= _NANOS_PER_SECOND
    self.seconds = seconds
    self.nanos = nanos
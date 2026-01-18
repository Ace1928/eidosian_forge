from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DurationValueValuesEnum(_messages.Enum):
    """From how long ago in the past these intervals were observed.

    Values:
      DURATION_UNSPECIFIED: <no description>
      HOUR: <no description>
      MAX: From BfdSession object creation time.
      MINUTE: <no description>
    """
    DURATION_UNSPECIFIED = 0
    HOUR = 1
    MAX = 2
    MINUTE = 3
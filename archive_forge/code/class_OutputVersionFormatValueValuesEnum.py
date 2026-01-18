from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OutputVersionFormatValueValuesEnum(_messages.Enum):
    """Deprecated. This field is unused.

    Values:
      VERSION_FORMAT_UNSPECIFIED: An unspecified format version that will
        default to V2.
      V2: LogEntry version 2 format.
      V1: LogEntry version 1 format.
    """
    VERSION_FORMAT_UNSPECIFIED = 0
    V2 = 1
    V1 = 2
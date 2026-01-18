from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SupportedEventsValueListEntryValuesEnum(_messages.Enum):
    """SupportedEventsValueListEntryValuesEnum enum type.

    Values:
      EVENT_TYPE_UNSPECIFIED: Unspecified value. Do not use.
      REQUEST_HEADERS: If included in `supported_events`, the HTTP request
        headers are processed.
      RESPONSE_HEADERS: If included in `supported_events`, the HTTP response
        headers are processed.
    """
    EVENT_TYPE_UNSPECIFIED = 0
    REQUEST_HEADERS = 1
    RESPONSE_HEADERS = 2
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GranularityValueValuesEnum(_messages.Enum):
    """Immutable. The granularity (i.e. `MILLIS`) at which timestamps are
    stored in this table. Timestamps not matching the granularity will be
    rejected. If unspecified at creation time, the value will be set to
    `MILLIS`. Views: `SCHEMA_VIEW`, `FULL`.

    Values:
      TIMESTAMP_GRANULARITY_UNSPECIFIED: The user did not specify a
        granularity. Should not be returned. When specified during table
        creation, MILLIS will be used.
      MILLIS: The table keeps data versioned at a granularity of 1ms.
    """
    TIMESTAMP_GRANULARITY_UNSPECIFIED = 0
    MILLIS = 1
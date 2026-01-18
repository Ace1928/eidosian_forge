from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WellKnownValueValuesEnum(_messages.Enum):
    """Well-known protobuf type such as `google.protobuf.Timestamp`.

    Values:
      WELL_KNOWN_TYPE_UNSPECIFIED: Unspecified type.
      ANY: Well-known protobuf.Any type. Any types are a polymorphic message
        type. During type-checking they are treated like `DYN` types, but at
        runtime they are resolved to a specific message type specified at
        evaluation time.
      TIMESTAMP: Well-known protobuf.Timestamp type, internally referenced as
        `timestamp`.
      DURATION: Well-known protobuf.Duration type, internally referenced as
        `duration`.
    """
    WELL_KNOWN_TYPE_UNSPECIFIED = 0
    ANY = 1
    TIMESTAMP = 2
    DURATION = 3
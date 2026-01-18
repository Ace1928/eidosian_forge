from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PayloadFormatValueValuesEnum(_messages.Enum):
    """Required. The desired format of the notification message payloads.

    Values:
      PAYLOAD_FORMAT_UNSPECIFIED: Illegal value, to avoid allowing a default.
      NONE: No payload is included with the notification.
      JSON: `TransferOperation` is [formatted as a JSON
        response](https://developers.google.com/protocol-
        buffers/docs/proto3#json), in application/json.
    """
    PAYLOAD_FORMAT_UNSPECIFIED = 0
    NONE = 1
    JSON = 2
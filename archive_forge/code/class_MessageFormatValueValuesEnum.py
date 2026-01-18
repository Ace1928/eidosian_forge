from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MessageFormatValueValuesEnum(_messages.Enum):
    """The format of the Cloud Pub/Sub messages.

    Values:
      MESSAGE_FORMAT_UNSPECIFIED: Unspecified.
      PROTOBUF: The message payload is a serialized protocol buffer of
        SourceRepoEvent.
      JSON: The message payload is a JSON string of SourceRepoEvent.
    """
    MESSAGE_FORMAT_UNSPECIFIED = 0
    PROTOBUF = 1
    JSON = 2
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VersioningValue(_messages.Message):
    """The bucket's versioning configuration.

    Fields:
      enabled: While set to true, versioning is fully enabled for this bucket.
    """
    enabled = _messages.BooleanField(1)
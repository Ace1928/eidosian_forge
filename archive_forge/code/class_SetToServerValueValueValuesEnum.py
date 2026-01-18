from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SetToServerValueValueValuesEnum(_messages.Enum):
    """Sets the field to the given server value.

    Values:
      SERVER_VALUE_UNSPECIFIED: Unspecified. This value must not be used.
      REQUEST_TIME: The time at which the server processed the request, with
        millisecond precision. If used on multiple fields (same or different
        documents) in a transaction, all the fields will get the same server
        timestamp.
    """
    SERVER_VALUE_UNSPECIFIED = 0
    REQUEST_TIME = 1
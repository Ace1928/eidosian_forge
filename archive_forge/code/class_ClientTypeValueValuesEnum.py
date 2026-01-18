from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClientTypeValueValuesEnum(_messages.Enum):
    """Immutable. The type of oauth client. either public or private.

    Values:
      CLIENT_TYPE_UNSPECIFIED: should not be used
      PUBLIC_CLIENT: public client has no secret
      CONFIDENTIAL_CLIENT: private client
    """
    CLIENT_TYPE_UNSPECIFIED = 0
    PUBLIC_CLIENT = 1
    CONFIDENTIAL_CLIENT = 2
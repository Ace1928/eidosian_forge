from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TransportProtocolValueValuesEnum(_messages.Enum):
    """Required. Immutable. The transport protocol used between the client
    and the server.

    Values:
      TRANSPORT_PROTOCOL_UNSPECIFIED: Default value. This value is unused.
      TCP: TCP protocol.
    """
    TRANSPORT_PROTOCOL_UNSPECIFIED = 0
    TCP = 1
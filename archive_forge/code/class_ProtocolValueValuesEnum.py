from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProtocolValueValuesEnum(_messages.Enum):
    """Volume protocol.

    Values:
      PROTOCOL_UNSPECIFIED: Unspecified value.
      PROTOCOL_FC: Fibre channel.
      PROTOCOL_NFS: Network file system.
    """
    PROTOCOL_UNSPECIFIED = 0
    PROTOCOL_FC = 1
    PROTOCOL_NFS = 2
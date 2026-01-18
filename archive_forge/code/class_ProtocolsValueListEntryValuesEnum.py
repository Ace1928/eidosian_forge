from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProtocolsValueListEntryValuesEnum(_messages.Enum):
    """ProtocolsValueListEntryValuesEnum enum type.

    Values:
      PROTOCOLS_UNSPECIFIED: Unspecified protocol
      NFSV3: NFS V3 protocol
      NFSV4: NFS V4 protocol
      SMB: SMB protocol
    """
    PROTOCOLS_UNSPECIFIED = 0
    NFSV3 = 1
    NFSV4 = 2
    SMB = 3
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ChannelCredentialTypeValueValuesEnum(_messages.Enum):
    """The channel credentials to access the SDS server. This field can be
    set to one of the following: CERTIFICATES: Use TLS certificates to access
    the SDS server. GCE_VM: Use local GCE VM credentials to access the SDS
    server.

    Values:
      CERTIFICATES: Use TLS certificates to access the SDS server.
      GCE_VM: Use local GCE VM credentials to access the SDS server.
      INVALID: <no description>
    """
    CERTIFICATES = 0
    GCE_VM = 1
    INVALID = 2
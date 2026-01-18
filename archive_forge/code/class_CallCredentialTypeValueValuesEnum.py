from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CallCredentialTypeValueValuesEnum(_messages.Enum):
    """The type of call credentials to use for GRPC requests to the SDS
    server. This field can be set to one of the following: - GCE_VM: The local
    GCE VM service account credentials are used to access the SDS server. -
    FROM_PLUGIN: Custom authenticator credentials are used to access the SDS
    server.

    Values:
      FROM_PLUGIN: Custom authenticator credentials are used to access the SDS
        server.
      GCE_VM: The local GCE VM service account credentials are used to access
        the SDS server.
      INVALID: <no description>
    """
    FROM_PLUGIN = 0
    GCE_VM = 1
    INVALID = 2
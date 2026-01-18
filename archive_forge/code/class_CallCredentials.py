from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CallCredentials(_messages.Message):
    """[Deprecated] gRPC call credentials to access the SDS server. gRPC call
  credentials to access the SDS server.

  Enums:
    CallCredentialTypeValueValuesEnum: The type of call credentials to use for
      GRPC requests to the SDS server. This field can be set to one of the
      following: - GCE_VM: The local GCE VM service account credentials are
      used to access the SDS server. - FROM_PLUGIN: Custom authenticator
      credentials are used to access the SDS server.

  Fields:
    callCredentialType: The type of call credentials to use for GRPC requests
      to the SDS server. This field can be set to one of the following: -
      GCE_VM: The local GCE VM service account credentials are used to access
      the SDS server. - FROM_PLUGIN: Custom authenticator credentials are used
      to access the SDS server.
    fromPlugin: Custom authenticator credentials. Valid if callCredentialType
      is FROM_PLUGIN.
  """

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
    callCredentialType = _messages.EnumField('CallCredentialTypeValueValuesEnum', 1)
    fromPlugin = _messages.MessageField('MetadataCredentialsFromPlugin', 2)
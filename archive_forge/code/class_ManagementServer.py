from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagementServer(_messages.Message):
    """ManagementServer describes a single BackupDR ManagementServer instance.

  Enums:
    StateValueValuesEnum: Output only. The ManagementServer state.
    TypeValueValuesEnum: Optional. The type of the ManagementServer resource.

  Messages:
    LabelsValue: Optional. Resource labels to represent user provided
      metadata. Labels currently defined: 1. migrate_from_go= If set to true,
      the MS is created in migration ready mode.

  Fields:
    baProxyUri: Output only. The hostname or ip address of the exposed AGM
      endpoints, used by BAs to connect to BA proxy.
    createTime: Output only. The time when the instance was created.
    description: Optional. The description of the ManagementServer instance
      (2048 characters or less).
    etag: Optional. Server specified ETag for the ManagementServer resource to
      prevent simultaneous updates from overwiting each other.
    labels: Optional. Resource labels to represent user provided metadata.
      Labels currently defined: 1. migrate_from_go= If set to true, the MS is
      created in migration ready mode.
    managementUri: Output only. The hostname or ip address of the exposed AGM
      endpoints, used by clients to connect to AGM/RD graphical user interface
      and APIs.
    name: Output only. Identifier. The resource name.
    networks: Required. VPC networks to which the ManagementServer instance is
      connected. For this version, only a single network is supported.
    oauth2ClientId: Output only. The OAuth 2.0 client id is required to make
      API calls to the BackupDR instance API of this ManagementServer. This is
      the value that should be provided in the 'aud' field of the OIDC ID
      Token (see openid specification https://openid.net/specs/openid-connect-
      core-1_0.html#IDToken).
    state: Output only. The ManagementServer state.
    type: Optional. The type of the ManagementServer resource.
    updateTime: Output only. The time when the instance was updated.
    workforceIdentityBasedManagementUri: Output only. The hostnames of the
      exposed AGM endpoints for both types of user i.e. 1p and 3p, used to
      connect AGM/RM UI.
    workforceIdentityBasedOauth2ClientId: Output only. The OAuth client IDs
      for both types of user i.e. 1p and 3p.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The ManagementServer state.

    Values:
      INSTANCE_STATE_UNSPECIFIED: State not set.
      CREATING: The instance is being created.
      READY: The instance has been created and is fully usable.
      UPDATING: The instance configuration is being updated. Certain kinds of
        updates may cause the instance to become unusable while the update is
        in progress.
      DELETING: The instance is being deleted.
      REPAIRING: The instance is being repaired and may be unstable.
      MAINTENANCE: Maintenance is being performed on this instance.
      ERROR: The instance is experiencing an issue and might be unusable. You
        can get further details from the statusMessage field of Instance
        resource.
    """
        INSTANCE_STATE_UNSPECIFIED = 0
        CREATING = 1
        READY = 2
        UPDATING = 3
        DELETING = 4
        REPAIRING = 5
        MAINTENANCE = 6
        ERROR = 7

    class TypeValueValuesEnum(_messages.Enum):
        """Optional. The type of the ManagementServer resource.

    Values:
      INSTANCE_TYPE_UNSPECIFIED: Instance type is not mentioned.
      BACKUP_RESTORE: Instance for backup and restore management (i.e., AGM).
    """
        INSTANCE_TYPE_UNSPECIFIED = 0
        BACKUP_RESTORE = 1

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Resource labels to represent user provided metadata. Labels
    currently defined: 1. migrate_from_go= If set to true, the MS is created
    in migration ready mode.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    baProxyUri = _messages.StringField(1, repeated=True)
    createTime = _messages.StringField(2)
    description = _messages.StringField(3)
    etag = _messages.StringField(4)
    labels = _messages.MessageField('LabelsValue', 5)
    managementUri = _messages.MessageField('ManagementURI', 6)
    name = _messages.StringField(7)
    networks = _messages.MessageField('NetworkConfig', 8, repeated=True)
    oauth2ClientId = _messages.StringField(9)
    state = _messages.EnumField('StateValueValuesEnum', 10)
    type = _messages.EnumField('TypeValueValuesEnum', 11)
    updateTime = _messages.StringField(12)
    workforceIdentityBasedManagementUri = _messages.MessageField('WorkforceIdentityBasedManagementURI', 13)
    workforceIdentityBasedOauth2ClientId = _messages.MessageField('WorkforceIdentityBasedOAuth2ClientID', 14)
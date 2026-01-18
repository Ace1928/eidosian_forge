from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AzureSourceDetails(_messages.Message):
    """AzureSourceDetails message describes a specific source details for the
  Azure source type.

  Enums:
    StateValueValuesEnum: Output only. State of the source as determined by
      the health check.

  Messages:
    MigrationResourcesUserTagsValue: User specified tags to add to every M2VM
      generated resource in Azure. These tags will be set in addition to the
      default tags that are set as part of the migration process. The tags
      must not begin with the reserved prefix `m4ce` or `m2vm`.

  Fields:
    azureLocation: Immutable. The Azure location (region) that the source VMs
      will be migrated from.
    clientSecretCreds: Azure Credentials using tenant ID, client ID and
      secret.
    error: Output only. Provides details on the state of the Source in case of
      an error.
    migrationResourcesUserTags: User specified tags to add to every M2VM
      generated resource in Azure. These tags will be set in addition to the
      default tags that are set as part of the migration process. The tags
      must not begin with the reserved prefix `m4ce` or `m2vm`.
    resourceGroupId: Output only. The ID of the Azure resource group that
      contains all resources related to the migration process of this source.
    state: Output only. State of the source as determined by the health check.
    subscriptionId: Immutable. Azure subscription ID.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of the source as determined by the health check.

    Values:
      STATE_UNSPECIFIED: The state is unknown. This is used for API
        compatibility only and is not used by the system.
      PENDING: The state was not sampled by the health checks yet.
      FAILED: The source is available but might not be usable yet due to
        invalid credentials or another reason. The error message will contain
        further details.
      ACTIVE: The source exists and its credentials were verified.
    """
        STATE_UNSPECIFIED = 0
        PENDING = 1
        FAILED = 2
        ACTIVE = 3

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MigrationResourcesUserTagsValue(_messages.Message):
        """User specified tags to add to every M2VM generated resource in Azure.
    These tags will be set in addition to the default tags that are set as
    part of the migration process. The tags must not begin with the reserved
    prefix `m4ce` or `m2vm`.

    Messages:
      AdditionalProperty: An additional property for a
        MigrationResourcesUserTagsValue object.

    Fields:
      additionalProperties: Additional properties of type
        MigrationResourcesUserTagsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a MigrationResourcesUserTagsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    azureLocation = _messages.StringField(1)
    clientSecretCreds = _messages.MessageField('ClientSecretCredentials', 2)
    error = _messages.MessageField('Status', 3)
    migrationResourcesUserTags = _messages.MessageField('MigrationResourcesUserTagsValue', 4)
    resourceGroupId = _messages.StringField(5)
    state = _messages.EnumField('StateValueValuesEnum', 6)
    subscriptionId = _messages.StringField(7)
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AwsSourceDetails(_messages.Message):
    """AwsSourceDetails message describes a specific source details for the AWS
  source type.

  Enums:
    StateValueValuesEnum: Output only. State of the source as determined by
      the health check.

  Messages:
    MigrationResourcesUserTagsValue: User specified tags to add to every M2VM
      generated resource in AWS. These tags will be set in addition to the
      default tags that are set as part of the migration process. The tags
      must not begin with the reserved prefix `m2vm`.

  Fields:
    accessKeyCreds: AWS Credentials using access key id and secret.
    awsRegion: Immutable. The AWS region that the source VMs will be migrated
      from.
    error: Output only. Provides details on the state of the Source in case of
      an error.
    inventorySecurityGroupNames: AWS security group names to limit the scope
      of the source inventory.
    inventoryTagList: AWS resource tags to limit the scope of the source
      inventory.
    migrationResourcesUserTags: User specified tags to add to every M2VM
      generated resource in AWS. These tags will be set in addition to the
      default tags that are set as part of the migration process. The tags
      must not begin with the reserved prefix `m2vm`.
    publicIp: Output only. The source's public IP. All communication initiated
      by this source will originate from this IP.
    state: Output only. State of the source as determined by the health check.
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
        """User specified tags to add to every M2VM generated resource in AWS.
    These tags will be set in addition to the default tags that are set as
    part of the migration process. The tags must not begin with the reserved
    prefix `m2vm`.

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
    accessKeyCreds = _messages.MessageField('AccessKeyCredentials', 1)
    awsRegion = _messages.StringField(2)
    error = _messages.MessageField('Status', 3)
    inventorySecurityGroupNames = _messages.StringField(4, repeated=True)
    inventoryTagList = _messages.MessageField('Tag', 5, repeated=True)
    migrationResourcesUserTags = _messages.MessageField('MigrationResourcesUserTagsValue', 6)
    publicIp = _messages.StringField(7)
    state = _messages.EnumField('StateValueValuesEnum', 8)
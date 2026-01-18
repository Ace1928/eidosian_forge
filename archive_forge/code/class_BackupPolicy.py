from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackupPolicy(_messages.Message):
    """Backup Policy.

  Enums:
    StateValueValuesEnum: Output only. The backup policy state.

  Messages:
    LabelsValue: Resource labels to represent user provided metadata.

  Fields:
    assignedVolumeCount: Output only. The total number of volumes assigned by
      this backup policy.
    createTime: Output only. The time when the backup policy was created.
    dailyBackupLimit: Number of daily backups to keep. Note that the minimum
      daily backup limit is 2.
    description: Description of the backup policy.
    enabled: If enabled, make backups automatically according to the
      schedules. This will be applied to all volumes that have this policy
      attached and enforced on volume level. If not specified, default is
      true.
    labels: Resource labels to represent user provided metadata.
    monthlyBackupLimit: Number of monthly backups to keep. Note that the sum
      of daily, weekly and monthly backups should be greater than 1.
    name: Identifier. The resource name of the backup policy. Format: `project
      s/{project_id}/locations/{location}/backupPolicies/{backup_policy_id}`.
    state: Output only. The backup policy state.
    weeklyBackupLimit: Number of weekly backups to keep. Note that the sum of
      daily, weekly and monthly backups should be greater than 1.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The backup policy state.

    Values:
      STATE_UNSPECIFIED: State not set.
      CREATING: BackupPolicy is being created.
      READY: BackupPolicy is available for use.
      DELETING: BackupPolicy is being deleted.
      ERROR: BackupPolicy is not valid and cannot be used.
      UPDATING: BackupPolicy is being updated.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        READY = 2
        DELETING = 3
        ERROR = 4
        UPDATING = 5

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Resource labels to represent user provided metadata.

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
    assignedVolumeCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    createTime = _messages.StringField(2)
    dailyBackupLimit = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    description = _messages.StringField(4)
    enabled = _messages.BooleanField(5)
    labels = _messages.MessageField('LabelsValue', 6)
    monthlyBackupLimit = _messages.IntegerField(7, variant=_messages.Variant.INT32)
    name = _messages.StringField(8)
    state = _messages.EnumField('StateValueValuesEnum', 9)
    weeklyBackupLimit = _messages.IntegerField(10, variant=_messages.Variant.INT32)
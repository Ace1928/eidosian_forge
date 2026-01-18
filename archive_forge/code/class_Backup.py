from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Backup(_messages.Message):
    """Message describing Backup object.

  Enums:
    BackupTypeValueValuesEnum:
    StateValueValuesEnum: Output only. The Backup resource instance state.

  Messages:
    LabelsValue: Optional. Resource labels to represent user provided
      metadata. No labels currently defined.

  Fields:
    backupApplianceBackupProperties: A BackupApplianceBackupProperties
      attribute.
    backupApplianceLocks: Optional. The list of BackupLocks taken by the
      accessor Backup Appliance.
    backupType: A BackupTypeValueValuesEnum attribute.
    computeInstanceBackupProperties: A ComputeInstanceBackupProperties
      attribute.
    consistencyTime: Output only. The point in time when this backup was
      captured from the source.
    createTime: Output only. The time when the instance was created.
    description: Output only. The description of the Backup instance (2048
      characters or less).
    enforcedRetentionEndTime: Optional. The backup can not be deleted before
      this time.
    etag: Optional. Server specified ETag to prevent updates from overwriting
      each other.
    expireTime: Optional. When this backup is automatically expired.
    gcpBackupPlanInfo: Output only. Configuration for a GCP resource.
    labels: Optional. Resource labels to represent user provided metadata. No
      labels currently defined.
    name: Output only. Name of the resource.
    resourceSizeBytes: Output only. source resource size in bytes at the time
      of the backup.
    serviceLocks: Output only. The list of BackupLocks taken by the service to
      prevent the deletion of the backup.
    state: Output only. The Backup resource instance state.
    updateTime: Output only. The time when the instance was updated.
  """

    class BackupTypeValueValuesEnum(_messages.Enum):
        """BackupTypeValueValuesEnum enum type.

    Values:
      BACKUP_TYPE_UNSPECIFIED: <no description>
      SCHEDULED: <no description>
      ON_DEMAND: <no description>
    """
        BACKUP_TYPE_UNSPECIFIED = 0
        SCHEDULED = 1
        ON_DEMAND = 2

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The Backup resource instance state.

    Values:
      STATE_UNSPECIFIED: State not set.
      CREATING: The backup is being created.
      ACTIVE: The backup has been created and is fully usable.
      DELETING: The backup is being deleted.
      ERROR: The backup is experiencing an issue and might be unusable.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        ACTIVE = 2
        DELETING = 3
        ERROR = 4

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Resource labels to represent user provided metadata. No
    labels currently defined.

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
    backupApplianceBackupProperties = _messages.MessageField('BackupApplianceBackupProperties', 1)
    backupApplianceLocks = _messages.MessageField('BackupLock', 2, repeated=True)
    backupType = _messages.EnumField('BackupTypeValueValuesEnum', 3)
    computeInstanceBackupProperties = _messages.MessageField('ComputeInstanceBackupProperties', 4)
    consistencyTime = _messages.StringField(5)
    createTime = _messages.StringField(6)
    description = _messages.StringField(7)
    enforcedRetentionEndTime = _messages.StringField(8)
    etag = _messages.StringField(9)
    expireTime = _messages.StringField(10)
    gcpBackupPlanInfo = _messages.MessageField('GCPBackupPlanInfo', 11)
    labels = _messages.MessageField('LabelsValue', 12)
    name = _messages.StringField(13)
    resourceSizeBytes = _messages.IntegerField(14)
    serviceLocks = _messages.MessageField('BackupLock', 15, repeated=True)
    state = _messages.EnumField('StateValueValuesEnum', 16)
    updateTime = _messages.StringField(17)
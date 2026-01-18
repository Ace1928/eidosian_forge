from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlInstancesDeleteRequest(_messages.Message):
    """A SqlInstancesDeleteRequest object.

  Fields:
    finalBackupDescription: The description of the final backup.
    finalBackupExpiryTime: Optional. Final Backup expiration time. Timestamp
      in UTC of when this resource is considered expired.
    finalBackupTtlDays: Optional. Retention period of the final backup.
    instance: Cloud SQL instance ID. This does not include the project ID.
    project: Project ID of the project that contains the instance to be
      deleted.
    skipFinalBackup: By default we opt in for creating final backup
  """
    finalBackupDescription = _messages.StringField(1)
    finalBackupExpiryTime = _messages.StringField(2)
    finalBackupTtlDays = _messages.IntegerField(3)
    instance = _messages.StringField(4, required=True)
    project = _messages.StringField(5, required=True)
    skipFinalBackup = _messages.BooleanField(6)
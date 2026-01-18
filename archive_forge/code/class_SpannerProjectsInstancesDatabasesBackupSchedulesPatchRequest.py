from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstancesDatabasesBackupSchedulesPatchRequest(_messages.Message):
    """A SpannerProjectsInstancesDatabasesBackupSchedulesPatchRequest object.

  Fields:
    backupSchedule: A BackupSchedule resource to be passed as the request
      body.
    name: Identifier. Output only for the CreateBackupSchedule operation.
      Required for the UpdateBackupSchedule operation. A globally unique
      identifier for the backup schedule which cannot be changed. Values are
      of the form
      `projects//instances//databases//backupSchedules/a-z*[a-z0-9]` The final
      segment of the name must be between 2 and 60 characters in length.
    updateMask: Required. A mask specifying which fields in the BackupSchedule
      resource should be updated. This mask is relative to the BackupSchedule
      resource, not to the request message. The field mask must always be
      specified; this prevents any future fields from being erased
      accidentally.
  """
    backupSchedule = _messages.MessageField('BackupSchedule', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)
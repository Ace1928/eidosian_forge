from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstancesDatabasesBackupSchedulesCreateRequest(_messages.Message):
    """A SpannerProjectsInstancesDatabasesBackupSchedulesCreateRequest object.

  Fields:
    backupSchedule: A BackupSchedule resource to be passed as the request
      body.
    backupScheduleId: Required. The Id to use for the backup schedule. The
      `backup_schedule_id` appended to `parent` forms the full backup schedule
      name of the form `projects//instances//databases//backupSchedules/`.
    parent: Required. The name of the database that this backup schedule
      applies to.
  """
    backupSchedule = _messages.MessageField('BackupSchedule', 1)
    backupScheduleId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
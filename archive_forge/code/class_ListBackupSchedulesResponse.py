from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListBackupSchedulesResponse(_messages.Message):
    """The response for ListBackupSchedules.

  Fields:
    backupSchedules: The list of backup schedules for a database.
    nextPageToken: `next_page_token` can be sent in a subsequent
      ListBackupSchedules call to fetch more of the schedules.
  """
    backupSchedules = _messages.MessageField('BackupSchedule', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
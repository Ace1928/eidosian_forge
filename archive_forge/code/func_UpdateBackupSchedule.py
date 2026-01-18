from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firestore import api_utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as ex
def UpdateBackupSchedule(project, database, backup_schedule, retention):
    """Updates a backup schedule.

  Args:
    project: the project of the database of the backup schedule, a string.
    database: the database id of the backup schedule, a string.
    backup_schedule: the backup to read, a string.
    retention: the retention of the backup schedule, an int. At what relative
      time in the future, compared to the creation time of the backup should the
      backup be deleted. The unit is seconds.

  Returns:
    a backup schedule.
  """
    messages = api_utils.GetMessages()
    backup_schedule_updates = messages.GoogleFirestoreAdminV1BackupSchedule()
    if retention:
        backup_schedule_updates.retention = api_utils.FormatDurationString(retention)
    return _GetBackupSchedulesService().Patch(messages.FirestoreProjectsDatabasesBackupSchedulesPatchRequest(name='projects/{}/databases/{}/backupSchedules/{}'.format(project, database, backup_schedule), googleFirestoreAdminV1BackupSchedule=backup_schedule_updates))
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firestore import api_utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as ex
def CreateBackupSchedule(project, database, retention, recurrence, day_of_week=None):
    """Creates a backup schedule.

  Args:
    project: the project of the database of the backup schedule, a string.
    database: the database id of the backup schedule, a string.
    retention: the retention of the backup schedule, an int. At what relative
      time in the future, compared to the creation time of the backup should the
      backup be deleted. The unit is seconds.
    recurrence: the recurrence of the backup schedule, a string. The valid
      values are: daily and weekly.
    day_of_week: day of week for weekly backup schdeule.

  Returns:
    a backup schedule.

  Raises:
    InvalidArgumentException: if recurrence is invalid.
    ConflictingArgumentsException: if recurrence is daily but day-of-week is
    provided.
    RequiredArgumentException: if recurrence is weekly but day-of-week is not
    provided.
  """
    messages = api_utils.GetMessages()
    backup_schedule = messages.GoogleFirestoreAdminV1BackupSchedule()
    backup_schedule.retention = api_utils.FormatDurationString(retention)
    if recurrence == 'daily':
        if day_of_week is not None:
            raise ex.ConflictingArgumentsException('--day-of-week', 'Cannot set day of week for daily backup schedules.')
        backup_schedule.dailyRecurrence = messages.GoogleFirestoreAdminV1DailyRecurrence()
    elif recurrence == 'weekly':
        if day_of_week is None:
            raise ex.RequiredArgumentException('--day-of-week', 'Day of week is required for weekly backup schedules, please use --day-of-week to specify this value')
        backup_schedule.weeklyRecurrence = messages.GoogleFirestoreAdminV1WeeklyRecurrence()
        backup_schedule.weeklyRecurrence.day = ConvertDayOfWeek(day_of_week)
    else:
        raise ex.InvalidArgumentException('--recurrence', 'invalid recurrence: {}. The available values are: `daily` and `weekly`.'.format(recurrence))
    return _GetBackupSchedulesService().Create(messages.FirestoreProjectsDatabasesBackupSchedulesCreateRequest(parent='projects/{}/databases/{}'.format(project, database), googleFirestoreAdminV1BackupSchedule=backup_schedule))
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firestore import api_utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as ex
def GetBackupSchedule(project, database, backup_schedule):
    """Gets a backup schedule.

  Args:
    project: the project of the database of the backup schedule, a string.
    database: the database id of the backup schedule, a string.
    backup_schedule: the backup schedule to read, a string.

  Returns:
    a backup schedule.
  """
    messages = api_utils.GetMessages()
    return _GetBackupSchedulesService().Get(messages.FirestoreProjectsDatabasesBackupSchedulesGetRequest(name='projects/{}/databases/{}/backupSchedules/{}'.format(project, database, backup_schedule)))
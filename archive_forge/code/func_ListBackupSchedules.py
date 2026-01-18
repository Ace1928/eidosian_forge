from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firestore import api_utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as ex
def ListBackupSchedules(project, database):
    """Lists backup schedules under a database.

  Args:
    project: the project of the database of the backup schedule, a string.
    database: the database id of the backup schedule, a string.

  Returns:
    a list of backup schedules.
  """
    messages = api_utils.GetMessages()
    return list(_GetBackupSchedulesService().List(messages.FirestoreProjectsDatabasesBackupSchedulesListRequest(parent='projects/{}/databases/{}'.format(project, database))).backupSchedules)
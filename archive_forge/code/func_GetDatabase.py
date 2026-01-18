from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firestore import api_utils
def GetDatabase(project, database):
    """Performs a Firestore Admin v1 Database Get.

  Args:
    project: the project id to get, a string.
    database: the database id to get, a string.

  Returns:
    a database.
  """
    messages = api_utils.GetMessages()
    return _GetDatabaseService().Get(messages.FirestoreProjectsDatabasesGetRequest(name='projects/{}/databases/{}'.format(project, database)))
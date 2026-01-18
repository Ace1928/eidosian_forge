from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firestore import api_utils
def ListDatabases(project):
    """Lists all Firestore databases under the project.

  Args:
    project: the project ID to list databases, a string.

  Returns:
    a List of Databases.
  """
    messages = api_utils.GetMessages()
    return list(_GetDatabaseService().List(messages.FirestoreProjectsDatabasesListRequest(parent='projects/{}'.format(project))).databases)
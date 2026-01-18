from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.firestore import api_utils
def CreateIndex(project, database, collection_id, index):
    """Performs a Firestore Admin v1 Index Creation.

  Args:
    project: the project of the database of the index, a string.
    database: the database id of the index, a string.
    collection_id: the current group of the index, a string.
    index: the index to create, a googleFirestoreAdminV1Index message.

  Returns:
    an Operation.
  """
    messages = api_utils.GetMessages()
    return _GetIndexService().Create(messages.FirestoreProjectsDatabasesCollectionGroupsIndexesCreateRequest(parent='projects/{}/databases/{}/collectionGroups/{}'.format(project, database, collection_id), googleFirestoreAdminV1Index=index))
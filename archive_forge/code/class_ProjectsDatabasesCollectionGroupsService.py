from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.firestore.v1 import firestore_v1_messages as messages
class ProjectsDatabasesCollectionGroupsService(base_api.BaseApiService):
    """Service class for the projects_databases_collectionGroups resource."""
    _NAME = 'projects_databases_collectionGroups'

    def __init__(self, client):
        super(FirestoreV1.ProjectsDatabasesCollectionGroupsService, self).__init__(client)
        self._upload_configs = {}
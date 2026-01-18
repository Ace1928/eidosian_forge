from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.firestore.v1 import firestore_v1_messages as messages
class ProjectsDatabasesCollectionGroupsIndexesService(base_api.BaseApiService):
    """Service class for the projects_databases_collectionGroups_indexes resource."""
    _NAME = 'projects_databases_collectionGroups_indexes'

    def __init__(self, client):
        super(FirestoreV1.ProjectsDatabasesCollectionGroupsIndexesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a composite index. This returns a google.longrunning.Operation which may be used to track the status of the creation. The metadata for the operation will be the type IndexOperationMetadata.

      Args:
        request: (FirestoreProjectsDatabasesCollectionGroupsIndexesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/databases/{databasesId}/collectionGroups/{collectionGroupsId}/indexes', http_method='POST', method_id='firestore.projects.databases.collectionGroups.indexes.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/indexes', request_field='googleFirestoreAdminV1Index', request_type_name='FirestoreProjectsDatabasesCollectionGroupsIndexesCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a composite index.

      Args:
        request: (FirestoreProjectsDatabasesCollectionGroupsIndexesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/databases/{databasesId}/collectionGroups/{collectionGroupsId}/indexes/{indexesId}', http_method='DELETE', method_id='firestore.projects.databases.collectionGroups.indexes.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='FirestoreProjectsDatabasesCollectionGroupsIndexesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a composite index.

      Args:
        request: (FirestoreProjectsDatabasesCollectionGroupsIndexesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleFirestoreAdminV1Index) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/databases/{databasesId}/collectionGroups/{collectionGroupsId}/indexes/{indexesId}', http_method='GET', method_id='firestore.projects.databases.collectionGroups.indexes.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='FirestoreProjectsDatabasesCollectionGroupsIndexesGetRequest', response_type_name='GoogleFirestoreAdminV1Index', supports_download=False)

    def List(self, request, global_params=None):
        """Lists composite indexes.

      Args:
        request: (FirestoreProjectsDatabasesCollectionGroupsIndexesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleFirestoreAdminV1ListIndexesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/databases/{databasesId}/collectionGroups/{collectionGroupsId}/indexes', http_method='GET', method_id='firestore.projects.databases.collectionGroups.indexes.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/indexes', request_field='', request_type_name='FirestoreProjectsDatabasesCollectionGroupsIndexesListRequest', response_type_name='GoogleFirestoreAdminV1ListIndexesResponse', supports_download=False)
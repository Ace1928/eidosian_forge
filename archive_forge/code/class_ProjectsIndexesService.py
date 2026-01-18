from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datastore.v1 import datastore_v1_messages as messages
class ProjectsIndexesService(base_api.BaseApiService):
    """Service class for the projects_indexes resource."""
    _NAME = 'projects_indexes'

    def __init__(self, client):
        super(DatastoreV1.ProjectsIndexesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates the specified index. A newly created index's initial state is `CREATING`. On completion of the returned google.longrunning.Operation, the state will be `READY`. If the index already exists, the call will return an `ALREADY_EXISTS` status. During index creation, the process could result in an error, in which case the index will move to the `ERROR` state. The process can be recovered by fixing the data that caused the error, removing the index with delete, then re-creating the index with create. Indexes with a single property cannot be created.

      Args:
        request: (GoogleDatastoreAdminV1Index) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='datastore.projects.indexes.create', ordered_params=['projectId'], path_params=['projectId'], query_params=[], relative_path='v1/projects/{projectId}/indexes', request_field='<request>', request_type_name='GoogleDatastoreAdminV1Index', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an existing index. An index can only be deleted if it is in a `READY` or `ERROR` state. On successful execution of the request, the index will be in a `DELETING` state. And on completion of the returned google.longrunning.Operation, the index will be removed. During index deletion, the process could result in an error, in which case the index will move to the `ERROR` state. The process can be recovered by fixing the data that caused the error, followed by calling delete again.

      Args:
        request: (DatastoreProjectsIndexesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='datastore.projects.indexes.delete', ordered_params=['projectId', 'indexId'], path_params=['indexId', 'projectId'], query_params=[], relative_path='v1/projects/{projectId}/indexes/{indexId}', request_field='', request_type_name='DatastoreProjectsIndexesDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an index.

      Args:
        request: (DatastoreProjectsIndexesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleDatastoreAdminV1Index) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='datastore.projects.indexes.get', ordered_params=['projectId', 'indexId'], path_params=['indexId', 'projectId'], query_params=[], relative_path='v1/projects/{projectId}/indexes/{indexId}', request_field='', request_type_name='DatastoreProjectsIndexesGetRequest', response_type_name='GoogleDatastoreAdminV1Index', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the indexes that match the specified filters. Datastore uses an eventually consistent query to fetch the list of indexes and may occasionally return stale results.

      Args:
        request: (DatastoreProjectsIndexesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleDatastoreAdminV1ListIndexesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='datastore.projects.indexes.list', ordered_params=['projectId'], path_params=['projectId'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/projects/{projectId}/indexes', request_field='', request_type_name='DatastoreProjectsIndexesListRequest', response_type_name='GoogleDatastoreAdminV1ListIndexesResponse', supports_download=False)
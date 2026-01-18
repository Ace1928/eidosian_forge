from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.spanner.v1 import spanner_v1_messages as messages
class ProjectsInstancesDatabasesOperationsService(base_api.BaseApiService):
    """Service class for the projects_instances_databases_operations resource."""
    _NAME = 'projects_instances_databases_operations'

    def __init__(self, client):
        super(SpannerV1.ProjectsInstancesDatabasesOperationsService, self).__init__(client)
        self._upload_configs = {}

    def Cancel(self, request, global_params=None):
        """Starts asynchronous cancellation on a long-running operation. The server makes a best effort to cancel the operation, but success is not guaranteed. If the server doesn't support this method, it returns `google.rpc.Code.UNIMPLEMENTED`. Clients can use Operations.GetOperation or other methods to check whether the cancellation succeeded or whether the operation completed despite cancellation. On successful cancellation, the operation is not deleted; instead, it becomes an operation with an Operation.error value with a google.rpc.Status.code of 1, corresponding to `Code.CANCELLED`.

      Args:
        request: (SpannerProjectsInstancesDatabasesOperationsCancelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Cancel')
        return self._RunMethod(config, request, global_params=global_params)
    Cancel.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/operations/{operationsId}:cancel', http_method='POST', method_id='spanner.projects.instances.databases.operations.cancel', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:cancel', request_field='', request_type_name='SpannerProjectsInstancesDatabasesOperationsCancelRequest', response_type_name='Empty', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a long-running operation. This method indicates that the client is no longer interested in the operation result. It does not cancel the operation. If the server doesn't support this method, it returns `google.rpc.Code.UNIMPLEMENTED`.

      Args:
        request: (SpannerProjectsInstancesDatabasesOperationsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/operations/{operationsId}', http_method='DELETE', method_id='spanner.projects.instances.databases.operations.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='SpannerProjectsInstancesDatabasesOperationsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the latest state of a long-running operation. Clients can use this method to poll the operation result at intervals as recommended by the API service.

      Args:
        request: (SpannerProjectsInstancesDatabasesOperationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/operations/{operationsId}', http_method='GET', method_id='spanner.projects.instances.databases.operations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='SpannerProjectsInstancesDatabasesOperationsGetRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists operations that match the specified filter in the request. If the server doesn't support this method, it returns `UNIMPLEMENTED`.

      Args:
        request: (SpannerProjectsInstancesDatabasesOperationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListOperationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/operations', http_method='GET', method_id='spanner.projects.instances.databases.operations.list', ordered_params=['name'], path_params=['name'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+name}', request_field='', request_type_name='SpannerProjectsInstancesDatabasesOperationsListRequest', response_type_name='ListOperationsResponse', supports_download=False)
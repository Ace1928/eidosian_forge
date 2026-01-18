from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1beta1 import aiplatform_v1beta1_messages as messages
class ProjectsLocationsMetadataStoresContextsOperationsService(base_api.BaseApiService):
    """Service class for the projects_locations_metadataStores_contexts_operations resource."""
    _NAME = 'projects_locations_metadataStores_contexts_operations'

    def __init__(self, client):
        super(AiplatformV1beta1.ProjectsLocationsMetadataStoresContextsOperationsService, self).__init__(client)
        self._upload_configs = {}

    def Cancel(self, request, global_params=None):
        """Starts asynchronous cancellation on a long-running operation. The server makes a best effort to cancel the operation, but success is not guaranteed. If the server doesn't support this method, it returns `google.rpc.Code.UNIMPLEMENTED`. Clients can use Operations.GetOperation or other methods to check whether the cancellation succeeded or whether the operation completed despite cancellation. On successful cancellation, the operation is not deleted; instead, it becomes an operation with an Operation.error value with a google.rpc.Status.code of 1, corresponding to `Code.CANCELLED`.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresContextsOperationsCancelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Cancel')
        return self._RunMethod(config, request, global_params=global_params)
    Cancel.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/metadataStores/{metadataStoresId}/contexts/{contextsId}/operations/{operationsId}:cancel', http_method='POST', method_id='aiplatform.projects.locations.metadataStores.contexts.operations.cancel', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}:cancel', request_field='', request_type_name='AiplatformProjectsLocationsMetadataStoresContextsOperationsCancelRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a long-running operation. This method indicates that the client is no longer interested in the operation result. It does not cancel the operation. If the server doesn't support this method, it returns `google.rpc.Code.UNIMPLEMENTED`.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresContextsOperationsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/metadataStores/{metadataStoresId}/contexts/{contextsId}/operations/{operationsId}', http_method='DELETE', method_id='aiplatform.projects.locations.metadataStores.contexts.operations.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsMetadataStoresContextsOperationsDeleteRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the latest state of a long-running operation. Clients can use this method to poll the operation result at intervals as recommended by the API service.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresContextsOperationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/metadataStores/{metadataStoresId}/contexts/{contextsId}/operations/{operationsId}', http_method='GET', method_id='aiplatform.projects.locations.metadataStores.contexts.operations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsMetadataStoresContextsOperationsGetRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists operations that match the specified filter in the request. If the server doesn't support this method, it returns `UNIMPLEMENTED`.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresContextsOperationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningListOperationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/metadataStores/{metadataStoresId}/contexts/{contextsId}/operations', http_method='GET', method_id='aiplatform.projects.locations.metadataStores.contexts.operations.list', ordered_params=['name'], path_params=['name'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1beta1/{+name}/operations', request_field='', request_type_name='AiplatformProjectsLocationsMetadataStoresContextsOperationsListRequest', response_type_name='GoogleLongrunningListOperationsResponse', supports_download=False)

    def Wait(self, request, global_params=None):
        """Waits until the specified long-running operation is done or reaches at most a specified timeout, returning the latest state. If the operation is already done, the latest state is immediately returned. If the timeout specified is greater than the default HTTP/RPC timeout, the HTTP/RPC timeout is used. If the server does not support this method, it returns `google.rpc.Code.UNIMPLEMENTED`. Note that this method is on a best-effort basis. It may return the latest state before the specified timeout (including immediately), meaning even an immediate response is no guarantee that the operation is done.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresContextsOperationsWaitRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Wait')
        return self._RunMethod(config, request, global_params=global_params)
    Wait.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/metadataStores/{metadataStoresId}/contexts/{contextsId}/operations/{operationsId}:wait', http_method='POST', method_id='aiplatform.projects.locations.metadataStores.contexts.operations.wait', ordered_params=['name'], path_params=['name'], query_params=['timeout'], relative_path='v1beta1/{+name}:wait', request_field='', request_type_name='AiplatformProjectsLocationsMetadataStoresContextsOperationsWaitRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)
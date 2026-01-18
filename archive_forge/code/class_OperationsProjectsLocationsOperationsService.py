from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.videointelligence.v1 import videointelligence_v1_messages as messages
class OperationsProjectsLocationsOperationsService(base_api.BaseApiService):
    """Service class for the operations_projects_locations_operations resource."""
    _NAME = 'operations_projects_locations_operations'

    def __init__(self, client):
        super(VideointelligenceV1.OperationsProjectsLocationsOperationsService, self).__init__(client)
        self._upload_configs = {}

    def Cancel(self, request, global_params=None):
        """Starts asynchronous cancellation on a long-running operation. The server makes a best effort to cancel the operation, but success is not guaranteed. If the server doesn't support this method, it returns `google.rpc.Code.UNIMPLEMENTED`. Clients can use Operations.GetOperation or other methods to check whether the cancellation succeeded or whether the operation completed despite cancellation. On successful cancellation, the operation is not deleted; instead, it becomes an operation with an Operation.error value with a google.rpc.Status.code of 1, corresponding to `Code.CANCELLED`.

      Args:
        request: (VideointelligenceOperationsProjectsLocationsOperationsCancelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Cancel')
        return self._RunMethod(config, request, global_params=global_params)
    Cancel.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/operations/projects/{projectsId}/locations/{locationsId}/operations/{operationsId}:cancel', http_method='POST', method_id='videointelligence.operations.projects.locations.operations.cancel', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/operations/{+name}:cancel', request_field='', request_type_name='VideointelligenceOperationsProjectsLocationsOperationsCancelRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a long-running operation. This method indicates that the client is no longer interested in the operation result. It does not cancel the operation. If the server doesn't support this method, it returns `google.rpc.Code.UNIMPLEMENTED`.

      Args:
        request: (VideointelligenceOperationsProjectsLocationsOperationsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/operations/projects/{projectsId}/locations/{locationsId}/operations/{operationsId}', http_method='DELETE', method_id='videointelligence.operations.projects.locations.operations.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/operations/{+name}', request_field='', request_type_name='VideointelligenceOperationsProjectsLocationsOperationsDeleteRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the latest state of a long-running operation. Clients can use this method to poll the operation result at intervals as recommended by the API service.

      Args:
        request: (VideointelligenceOperationsProjectsLocationsOperationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/operations/projects/{projectsId}/locations/{locationsId}/operations/{operationsId}', http_method='GET', method_id='videointelligence.operations.projects.locations.operations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/operations/{+name}', request_field='', request_type_name='VideointelligenceOperationsProjectsLocationsOperationsGetRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)
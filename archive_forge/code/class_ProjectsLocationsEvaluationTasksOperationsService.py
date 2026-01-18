from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1beta1 import aiplatform_v1beta1_messages as messages
class ProjectsLocationsEvaluationTasksOperationsService(base_api.BaseApiService):
    """Service class for the projects_locations_evaluationTasks_operations resource."""
    _NAME = 'projects_locations_evaluationTasks_operations'

    def __init__(self, client):
        super(AiplatformV1beta1.ProjectsLocationsEvaluationTasksOperationsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes a long-running operation. This method indicates that the client is no longer interested in the operation result. It does not cancel the operation. If the server doesn't support this method, it returns `google.rpc.Code.UNIMPLEMENTED`.

      Args:
        request: (AiplatformProjectsLocationsEvaluationTasksOperationsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/evaluationTasks/{evaluationTasksId}/operations/{operationsId}', http_method='DELETE', method_id='aiplatform.projects.locations.evaluationTasks.operations.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsEvaluationTasksOperationsDeleteRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the latest state of a long-running operation. Clients can use this method to poll the operation result at intervals as recommended by the API service.

      Args:
        request: (AiplatformProjectsLocationsEvaluationTasksOperationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/evaluationTasks/{evaluationTasksId}/operations/{operationsId}', http_method='GET', method_id='aiplatform.projects.locations.evaluationTasks.operations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsEvaluationTasksOperationsGetRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists operations that match the specified filter in the request. If the server doesn't support this method, it returns `UNIMPLEMENTED`.

      Args:
        request: (AiplatformProjectsLocationsEvaluationTasksOperationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningListOperationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/evaluationTasks/{evaluationTasksId}/operations', http_method='GET', method_id='aiplatform.projects.locations.evaluationTasks.operations.list', ordered_params=['name'], path_params=['name'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1beta1/{+name}/operations', request_field='', request_type_name='AiplatformProjectsLocationsEvaluationTasksOperationsListRequest', response_type_name='GoogleLongrunningListOperationsResponse', supports_download=False)

    def Wait(self, request, global_params=None):
        """Waits until the specified long-running operation is done or reaches at most a specified timeout, returning the latest state. If the operation is already done, the latest state is immediately returned. If the timeout specified is greater than the default HTTP/RPC timeout, the HTTP/RPC timeout is used. If the server does not support this method, it returns `google.rpc.Code.UNIMPLEMENTED`. Note that this method is on a best-effort basis. It may return the latest state before the specified timeout (including immediately), meaning even an immediate response is no guarantee that the operation is done.

      Args:
        request: (AiplatformProjectsLocationsEvaluationTasksOperationsWaitRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Wait')
        return self._RunMethod(config, request, global_params=global_params)
    Wait.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/evaluationTasks/{evaluationTasksId}/operations/{operationsId}:wait', http_method='POST', method_id='aiplatform.projects.locations.evaluationTasks.operations.wait', ordered_params=['name'], path_params=['name'], query_params=['timeout'], relative_path='v1beta1/{+name}:wait', request_field='', request_type_name='AiplatformProjectsLocationsEvaluationTasksOperationsWaitRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)
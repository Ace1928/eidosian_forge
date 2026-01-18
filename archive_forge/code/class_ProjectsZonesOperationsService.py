from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.container.v1 import container_v1_messages as messages
class ProjectsZonesOperationsService(base_api.BaseApiService):
    """Service class for the projects_zones_operations resource."""
    _NAME = 'projects_zones_operations'

    def __init__(self, client):
        super(ContainerV1.ProjectsZonesOperationsService, self).__init__(client)
        self._upload_configs = {}

    def Cancel(self, request, global_params=None):
        """Cancels the specified operation.

      Args:
        request: (CancelOperationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Cancel')
        return self._RunMethod(config, request, global_params=global_params)
    Cancel.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='container.projects.zones.operations.cancel', ordered_params=['projectId', 'zone', 'operationId'], path_params=['operationId', 'projectId', 'zone'], query_params=[], relative_path='v1/projects/{projectId}/zones/{zone}/operations/{operationId}:cancel', request_field='<request>', request_type_name='CancelOperationRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the specified operation.

      Args:
        request: (ContainerProjectsZonesOperationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='container.projects.zones.operations.get', ordered_params=['projectId', 'zone', 'operationId'], path_params=['operationId', 'projectId', 'zone'], query_params=['name'], relative_path='v1/projects/{projectId}/zones/{zone}/operations/{operationId}', request_field='', request_type_name='ContainerProjectsZonesOperationsGetRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all operations in a project in a specific zone or all zones.

      Args:
        request: (ContainerProjectsZonesOperationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListOperationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='container.projects.zones.operations.list', ordered_params=['projectId', 'zone'], path_params=['projectId', 'zone'], query_params=['parent'], relative_path='v1/projects/{projectId}/zones/{zone}/operations', request_field='', request_type_name='ContainerProjectsZonesOperationsListRequest', response_type_name='ListOperationsResponse', supports_download=False)
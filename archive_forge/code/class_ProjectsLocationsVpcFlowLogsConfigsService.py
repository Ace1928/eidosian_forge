from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networkmanagement.v1beta1 import networkmanagement_v1beta1_messages as messages
class ProjectsLocationsVpcFlowLogsConfigsService(base_api.BaseApiService):
    """Service class for the projects_locations_vpcFlowLogsConfigs resource."""
    _NAME = 'projects_locations_vpcFlowLogsConfigs'

    def __init__(self, client):
        super(NetworkmanagementV1beta1.ProjectsLocationsVpcFlowLogsConfigsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new VPC Flow Logs configuration. If a configuration with the exact same settings already exists, the creation fails.

      Args:
        request: (NetworkmanagementProjectsLocationsVpcFlowLogsConfigsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/vpcFlowLogsConfigs', http_method='POST', method_id='networkmanagement.projects.locations.vpcFlowLogsConfigs.create', ordered_params=['parent'], path_params=['parent'], query_params=['vpcFlowLogsConfigId'], relative_path='v1beta1/{+parent}/vpcFlowLogsConfigs', request_field='vpcFlowLogsConfig', request_type_name='NetworkmanagementProjectsLocationsVpcFlowLogsConfigsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a specific VpcFlowLog configuration.

      Args:
        request: (NetworkmanagementProjectsLocationsVpcFlowLogsConfigsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/vpcFlowLogsConfigs/{vpcFlowLogsConfigsId}', http_method='DELETE', method_id='networkmanagement.projects.locations.vpcFlowLogsConfigs.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}', request_field='', request_type_name='NetworkmanagementProjectsLocationsVpcFlowLogsConfigsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the details of a specific VPC Flow Log configuration.

      Args:
        request: (NetworkmanagementProjectsLocationsVpcFlowLogsConfigsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (VpcFlowLogsConfig) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/vpcFlowLogsConfigs/{vpcFlowLogsConfigsId}', http_method='GET', method_id='networkmanagement.projects.locations.vpcFlowLogsConfigs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}', request_field='', request_type_name='NetworkmanagementProjectsLocationsVpcFlowLogsConfigsGetRequest', response_type_name='VpcFlowLogsConfig', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all VPC Flow Logs configurations in a given project.

      Args:
        request: (NetworkmanagementProjectsLocationsVpcFlowLogsConfigsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListVpcFlowLogsConfigsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/vpcFlowLogsConfigs', http_method='GET', method_id='networkmanagement.projects.locations.vpcFlowLogsConfigs.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1beta1/{+parent}/vpcFlowLogsConfigs', request_field='', request_type_name='NetworkmanagementProjectsLocationsVpcFlowLogsConfigsListRequest', response_type_name='ListVpcFlowLogsConfigsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an existing VpcFlowLogsConfig. If a configuration with the exact same settings already exists, the creation fails.

      Args:
        request: (NetworkmanagementProjectsLocationsVpcFlowLogsConfigsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/vpcFlowLogsConfigs/{vpcFlowLogsConfigsId}', http_method='PATCH', method_id='networkmanagement.projects.locations.vpcFlowLogsConfigs.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1beta1/{+name}', request_field='vpcFlowLogsConfig', request_type_name='NetworkmanagementProjectsLocationsVpcFlowLogsConfigsPatchRequest', response_type_name='Operation', supports_download=False)
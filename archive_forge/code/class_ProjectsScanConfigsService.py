from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.containeranalysis.v1alpha1 import containeranalysis_v1alpha1_messages as messages
class ProjectsScanConfigsService(base_api.BaseApiService):
    """Service class for the projects_scanConfigs resource."""
    _NAME = 'projects_scanConfigs'

    def __init__(self, client):
        super(ContaineranalysisV1alpha1.ProjectsScanConfigsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets a specific scan configuration for a project.

      Args:
        request: (ContaineranalysisProjectsScanConfigsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ScanConfig) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/scanConfigs/{scanConfigsId}', http_method='GET', method_id='containeranalysis.projects.scanConfigs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='ContaineranalysisProjectsScanConfigsGetRequest', response_type_name='ScanConfig', supports_download=False)

    def List(self, request, global_params=None):
        """Lists scan configurations for a project.

      Args:
        request: (ContaineranalysisProjectsScanConfigsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListScanConfigsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/scanConfigs', http_method='GET', method_id='containeranalysis.projects.scanConfigs.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/scanConfigs', request_field='', request_type_name='ContaineranalysisProjectsScanConfigsListRequest', response_type_name='ListScanConfigsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the scan configuration to a new value.

      Args:
        request: (ContaineranalysisProjectsScanConfigsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ScanConfig) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/scanConfigs/{scanConfigsId}', http_method='PATCH', method_id='containeranalysis.projects.scanConfigs.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha1/{+name}', request_field='scanConfig', request_type_name='ContaineranalysisProjectsScanConfigsPatchRequest', response_type_name='ScanConfig', supports_download=False)
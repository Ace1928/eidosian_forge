from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.monitoring.v3 import monitoring_v3_messages as messages
class ProjectsUptimeCheckConfigsService(base_api.BaseApiService):
    """Service class for the projects_uptimeCheckConfigs resource."""
    _NAME = 'projects_uptimeCheckConfigs'

    def __init__(self, client):
        super(MonitoringV3.ProjectsUptimeCheckConfigsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new Uptime check configuration.

      Args:
        request: (MonitoringProjectsUptimeCheckConfigsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (UptimeCheckConfig) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/uptimeCheckConfigs', http_method='POST', method_id='monitoring.projects.uptimeCheckConfigs.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v3/{+parent}/uptimeCheckConfigs', request_field='uptimeCheckConfig', request_type_name='MonitoringProjectsUptimeCheckConfigsCreateRequest', response_type_name='UptimeCheckConfig', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an Uptime check configuration. Note that this method will fail if the Uptime check configuration is referenced by an alert policy or other dependent configs that would be rendered invalid by the deletion.

      Args:
        request: (MonitoringProjectsUptimeCheckConfigsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/uptimeCheckConfigs/{uptimeCheckConfigsId}', http_method='DELETE', method_id='monitoring.projects.uptimeCheckConfigs.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v3/{+name}', request_field='', request_type_name='MonitoringProjectsUptimeCheckConfigsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a single Uptime check configuration.

      Args:
        request: (MonitoringProjectsUptimeCheckConfigsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (UptimeCheckConfig) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/uptimeCheckConfigs/{uptimeCheckConfigsId}', http_method='GET', method_id='monitoring.projects.uptimeCheckConfigs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v3/{+name}', request_field='', request_type_name='MonitoringProjectsUptimeCheckConfigsGetRequest', response_type_name='UptimeCheckConfig', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the existing valid Uptime check configurations for the project (leaving out any invalid configurations).

      Args:
        request: (MonitoringProjectsUptimeCheckConfigsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListUptimeCheckConfigsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/uptimeCheckConfigs', http_method='GET', method_id='monitoring.projects.uptimeCheckConfigs.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v3/{+parent}/uptimeCheckConfigs', request_field='', request_type_name='MonitoringProjectsUptimeCheckConfigsListRequest', response_type_name='ListUptimeCheckConfigsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an Uptime check configuration. You can either replace the entire configuration with a new one or replace only certain fields in the current configuration by specifying the fields to be updated via updateMask. Returns the updated configuration.

      Args:
        request: (MonitoringProjectsUptimeCheckConfigsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (UptimeCheckConfig) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/uptimeCheckConfigs/{uptimeCheckConfigsId}', http_method='PATCH', method_id='monitoring.projects.uptimeCheckConfigs.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v3/{+name}', request_field='uptimeCheckConfig', request_type_name='MonitoringProjectsUptimeCheckConfigsPatchRequest', response_type_name='UptimeCheckConfig', supports_download=False)
from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.monitoring.v3 import monitoring_v3_messages as messages
class ProjectsMonitoredResourceDescriptorsService(base_api.BaseApiService):
    """Service class for the projects_monitoredResourceDescriptors resource."""
    _NAME = 'projects_monitoredResourceDescriptors'

    def __init__(self, client):
        super(MonitoringV3.ProjectsMonitoredResourceDescriptorsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets a single monitored resource descriptor.

      Args:
        request: (MonitoringProjectsMonitoredResourceDescriptorsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (MonitoredResourceDescriptor) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/monitoredResourceDescriptors/{monitoredResourceDescriptorsId}', http_method='GET', method_id='monitoring.projects.monitoredResourceDescriptors.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v3/{+name}', request_field='', request_type_name='MonitoringProjectsMonitoredResourceDescriptorsGetRequest', response_type_name='MonitoredResourceDescriptor', supports_download=False)

    def List(self, request, global_params=None):
        """Lists monitored resource descriptors that match a filter.

      Args:
        request: (MonitoringProjectsMonitoredResourceDescriptorsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListMonitoredResourceDescriptorsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v3/projects/{projectsId}/monitoredResourceDescriptors', http_method='GET', method_id='monitoring.projects.monitoredResourceDescriptors.list', ordered_params=['name'], path_params=['name'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v3/{+name}/monitoredResourceDescriptors', request_field='', request_type_name='MonitoringProjectsMonitoredResourceDescriptorsListRequest', response_type_name='ListMonitoredResourceDescriptorsResponse', supports_download=False)
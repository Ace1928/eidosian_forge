from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.appconfigmanager.v1alpha import appconfigmanager_v1alpha_messages as messages
class ProjectsLocationsConfigsVersionRendersService(base_api.BaseApiService):
    """Service class for the projects_locations_configs_versionRenders resource."""
    _NAME = 'projects_locations_configs_versionRenders'

    def __init__(self, client):
        super(AppconfigmanagerV1alpha.ProjectsLocationsConfigsVersionRendersService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets details of a single ConfigVersionRender.

      Args:
        request: (AppconfigmanagerProjectsLocationsConfigsVersionRendersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ConfigVersionRender) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/configs/{configsId}/versionRenders/{versionRendersId}', http_method='GET', method_id='appconfigmanager.projects.locations.configs.versionRenders.get', ordered_params=['name'], path_params=['name'], query_params=['view'], relative_path='v1alpha/{+name}', request_field='', request_type_name='AppconfigmanagerProjectsLocationsConfigsVersionRendersGetRequest', response_type_name='ConfigVersionRender', supports_download=False)

    def List(self, request, global_params=None):
        """Lists ConfigVersionRenders in a given project, location, and Config.

      Args:
        request: (AppconfigmanagerProjectsLocationsConfigsVersionRendersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListConfigVersionRendersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/configs/{configsId}/versionRenders', http_method='GET', method_id='appconfigmanager.projects.locations.configs.versionRenders.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken', 'view'], relative_path='v1alpha/{+parent}/versionRenders', request_field='', request_type_name='AppconfigmanagerProjectsLocationsConfigsVersionRendersListRequest', response_type_name='ListConfigVersionRendersResponse', supports_download=False)
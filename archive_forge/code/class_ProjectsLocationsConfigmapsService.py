from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.anthosevents.v1 import anthosevents_v1_messages as messages
class ProjectsLocationsConfigmapsService(base_api.BaseApiService):
    """Service class for the projects_locations_configmaps resource."""
    _NAME = 'projects_locations_configmaps'

    def __init__(self, client):
        super(AnthoseventsV1.ProjectsLocationsConfigmapsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new config map.

      Args:
        request: (AnthoseventsProjectsLocationsConfigmapsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ConfigMap) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/configmaps', http_method='POST', method_id='anthosevents.projects.locations.configmaps.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/configmaps', request_field='configMap', request_type_name='AnthoseventsProjectsLocationsConfigmapsCreateRequest', response_type_name='ConfigMap', supports_download=False)

    def Get(self, request, global_params=None):
        """Rpc to retrieve config map.

      Args:
        request: (AnthoseventsProjectsLocationsConfigmapsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ConfigMap) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/configmaps/{configmapsId}', http_method='GET', method_id='anthosevents.projects.locations.configmaps.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AnthoseventsProjectsLocationsConfigmapsGetRequest', response_type_name='ConfigMap', supports_download=False)
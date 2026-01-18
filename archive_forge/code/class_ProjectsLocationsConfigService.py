from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.speech.v2 import speech_v2_messages as messages
class ProjectsLocationsConfigService(base_api.BaseApiService):
    """Service class for the projects_locations_config resource."""
    _NAME = 'projects_locations_config'

    def __init__(self, client):
        super(SpeechV2.ProjectsLocationsConfigService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Returns the requested Config.

      Args:
        request: (SpeechProjectsLocationsConfigGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Config) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/config', http_method='GET', method_id='speech.projects.locations.config.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='SpeechProjectsLocationsConfigGetRequest', response_type_name='Config', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates the Config.

      Args:
        request: (SpeechProjectsLocationsConfigUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Config) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/config', http_method='PATCH', method_id='speech.projects.locations.config.update', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}', request_field='config', request_type_name='SpeechProjectsLocationsConfigUpdateRequest', response_type_name='Config', supports_download=False)
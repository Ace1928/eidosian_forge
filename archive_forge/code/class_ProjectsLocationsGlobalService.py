from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.osconfig.v1 import osconfig_v1_messages as messages
class ProjectsLocationsGlobalService(base_api.BaseApiService):
    """Service class for the projects_locations_global resource."""
    _NAME = 'projects_locations_global'

    def __init__(self, client):
        super(OsconfigV1.ProjectsLocationsGlobalService, self).__init__(client)
        self._upload_configs = {}

    def GetProjectFeatureSettings(self, request, global_params=None):
        """GetProjectFeatureSettings returns the feature settings for a project.

      Args:
        request: (OsconfigProjectsLocationsGlobalGetProjectFeatureSettingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ProjectFeatureSettings) The response message.
      """
        config = self.GetMethodConfig('GetProjectFeatureSettings')
        return self._RunMethod(config, request, global_params=global_params)
    GetProjectFeatureSettings.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/projectFeatureSettings', http_method='GET', method_id='osconfig.projects.locations.global.getProjectFeatureSettings', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='OsconfigProjectsLocationsGlobalGetProjectFeatureSettingsRequest', response_type_name='ProjectFeatureSettings', supports_download=False)

    def UpdateProjectFeatureSettings(self, request, global_params=None):
        """UpdateProjectFeatureSettings sets the feature settings for a project.

      Args:
        request: (OsconfigProjectsLocationsGlobalUpdateProjectFeatureSettingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ProjectFeatureSettings) The response message.
      """
        config = self.GetMethodConfig('UpdateProjectFeatureSettings')
        return self._RunMethod(config, request, global_params=global_params)
    UpdateProjectFeatureSettings.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/projectFeatureSettings', http_method='PATCH', method_id='osconfig.projects.locations.global.updateProjectFeatureSettings', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='projectFeatureSettings', request_type_name='OsconfigProjectsLocationsGlobalUpdateProjectFeatureSettingsRequest', response_type_name='ProjectFeatureSettings', supports_download=False)
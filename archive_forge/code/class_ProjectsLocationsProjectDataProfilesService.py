from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dlp.v2 import dlp_v2_messages as messages
class ProjectsLocationsProjectDataProfilesService(base_api.BaseApiService):
    """Service class for the projects_locations_projectDataProfiles resource."""
    _NAME = 'projects_locations_projectDataProfiles'

    def __init__(self, client):
        super(DlpV2.ProjectsLocationsProjectDataProfilesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets a project data profile.

      Args:
        request: (DlpProjectsLocationsProjectDataProfilesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2ProjectDataProfile) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/projectDataProfiles/{projectDataProfilesId}', http_method='GET', method_id='dlp.projects.locations.projectDataProfiles.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DlpProjectsLocationsProjectDataProfilesGetRequest', response_type_name='GooglePrivacyDlpV2ProjectDataProfile', supports_download=False)

    def List(self, request, global_params=None):
        """Lists project data profiles for an organization.

      Args:
        request: (DlpProjectsLocationsProjectDataProfilesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2ListProjectDataProfilesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/projectDataProfiles', http_method='GET', method_id='dlp.projects.locations.projectDataProfiles.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v2/{+parent}/projectDataProfiles', request_field='', request_type_name='DlpProjectsLocationsProjectDataProfilesListRequest', response_type_name='GooglePrivacyDlpV2ListProjectDataProfilesResponse', supports_download=False)
from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dlp.v2 import dlp_v2_messages as messages
class ProjectsLocationsColumnDataProfilesService(base_api.BaseApiService):
    """Service class for the projects_locations_columnDataProfiles resource."""
    _NAME = 'projects_locations_columnDataProfiles'

    def __init__(self, client):
        super(DlpV2.ProjectsLocationsColumnDataProfilesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets a column data profile.

      Args:
        request: (DlpProjectsLocationsColumnDataProfilesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2ColumnDataProfile) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/columnDataProfiles/{columnDataProfilesId}', http_method='GET', method_id='dlp.projects.locations.columnDataProfiles.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DlpProjectsLocationsColumnDataProfilesGetRequest', response_type_name='GooglePrivacyDlpV2ColumnDataProfile', supports_download=False)

    def List(self, request, global_params=None):
        """Lists column data profiles for an organization.

      Args:
        request: (DlpProjectsLocationsColumnDataProfilesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2ListColumnDataProfilesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/columnDataProfiles', http_method='GET', method_id='dlp.projects.locations.columnDataProfiles.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v2/{+parent}/columnDataProfiles', request_field='', request_type_name='DlpProjectsLocationsColumnDataProfilesListRequest', response_type_name='GooglePrivacyDlpV2ListColumnDataProfilesResponse', supports_download=False)
from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dlp.v2 import dlp_v2_messages as messages
class ProjectsLocationsTableDataProfilesService(base_api.BaseApiService):
    """Service class for the projects_locations_tableDataProfiles resource."""
    _NAME = 'projects_locations_tableDataProfiles'

    def __init__(self, client):
        super(DlpV2.ProjectsLocationsTableDataProfilesService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Delete a TableDataProfile. Will not prevent the profile from being regenerated if the table is still included in a discovery configuration.

      Args:
        request: (DlpProjectsLocationsTableDataProfilesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/tableDataProfiles/{tableDataProfilesId}', http_method='DELETE', method_id='dlp.projects.locations.tableDataProfiles.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DlpProjectsLocationsTableDataProfilesDeleteRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a table data profile.

      Args:
        request: (DlpProjectsLocationsTableDataProfilesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2TableDataProfile) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/tableDataProfiles/{tableDataProfilesId}', http_method='GET', method_id='dlp.projects.locations.tableDataProfiles.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DlpProjectsLocationsTableDataProfilesGetRequest', response_type_name='GooglePrivacyDlpV2TableDataProfile', supports_download=False)

    def List(self, request, global_params=None):
        """Lists table data profiles for an organization.

      Args:
        request: (DlpProjectsLocationsTableDataProfilesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2ListTableDataProfilesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/tableDataProfiles', http_method='GET', method_id='dlp.projects.locations.tableDataProfiles.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v2/{+parent}/tableDataProfiles', request_field='', request_type_name='DlpProjectsLocationsTableDataProfilesListRequest', response_type_name='GooglePrivacyDlpV2ListTableDataProfilesResponse', supports_download=False)
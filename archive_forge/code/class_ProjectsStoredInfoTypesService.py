from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dlp.v2 import dlp_v2_messages as messages
class ProjectsStoredInfoTypesService(base_api.BaseApiService):
    """Service class for the projects_storedInfoTypes resource."""
    _NAME = 'projects_storedInfoTypes'

    def __init__(self, client):
        super(DlpV2.ProjectsStoredInfoTypesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a pre-built stored infoType to be used for inspection. See https://cloud.google.com/sensitive-data-protection/docs/creating-stored-infotypes to learn more.

      Args:
        request: (DlpProjectsStoredInfoTypesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2StoredInfoType) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/storedInfoTypes', http_method='POST', method_id='dlp.projects.storedInfoTypes.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/storedInfoTypes', request_field='googlePrivacyDlpV2CreateStoredInfoTypeRequest', request_type_name='DlpProjectsStoredInfoTypesCreateRequest', response_type_name='GooglePrivacyDlpV2StoredInfoType', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a stored infoType. See https://cloud.google.com/sensitive-data-protection/docs/creating-stored-infotypes to learn more.

      Args:
        request: (DlpProjectsStoredInfoTypesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/storedInfoTypes/{storedInfoTypesId}', http_method='DELETE', method_id='dlp.projects.storedInfoTypes.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DlpProjectsStoredInfoTypesDeleteRequest', response_type_name='GoogleProtobufEmpty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a stored infoType. See https://cloud.google.com/sensitive-data-protection/docs/creating-stored-infotypes to learn more.

      Args:
        request: (DlpProjectsStoredInfoTypesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2StoredInfoType) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/storedInfoTypes/{storedInfoTypesId}', http_method='GET', method_id='dlp.projects.storedInfoTypes.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='DlpProjectsStoredInfoTypesGetRequest', response_type_name='GooglePrivacyDlpV2StoredInfoType', supports_download=False)

    def List(self, request, global_params=None):
        """Lists stored infoTypes. See https://cloud.google.com/sensitive-data-protection/docs/creating-stored-infotypes to learn more.

      Args:
        request: (DlpProjectsStoredInfoTypesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2ListStoredInfoTypesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/storedInfoTypes', http_method='GET', method_id='dlp.projects.storedInfoTypes.list', ordered_params=['parent'], path_params=['parent'], query_params=['locationId', 'orderBy', 'pageSize', 'pageToken'], relative_path='v2/{+parent}/storedInfoTypes', request_field='', request_type_name='DlpProjectsStoredInfoTypesListRequest', response_type_name='GooglePrivacyDlpV2ListStoredInfoTypesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the stored infoType by creating a new version. The existing version will continue to be used until the new version is ready. See https://cloud.google.com/sensitive-data-protection/docs/creating-stored-infotypes to learn more.

      Args:
        request: (DlpProjectsStoredInfoTypesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GooglePrivacyDlpV2StoredInfoType) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/storedInfoTypes/{storedInfoTypesId}', http_method='PATCH', method_id='dlp.projects.storedInfoTypes.patch', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='googlePrivacyDlpV2UpdateStoredInfoTypeRequest', request_type_name='DlpProjectsStoredInfoTypesPatchRequest', response_type_name='GooglePrivacyDlpV2StoredInfoType', supports_download=False)
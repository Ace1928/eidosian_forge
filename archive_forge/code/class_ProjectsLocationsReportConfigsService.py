from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.storageinsights.v1 import storageinsights_v1_messages as messages
class ProjectsLocationsReportConfigsService(base_api.BaseApiService):
    """Service class for the projects_locations_reportConfigs resource."""
    _NAME = 'projects_locations_reportConfigs'

    def __init__(self, client):
        super(StorageinsightsV1.ProjectsLocationsReportConfigsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new ReportConfig in a given project and location.

      Args:
        request: (StorageinsightsProjectsLocationsReportConfigsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ReportConfig) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/reportConfigs', http_method='POST', method_id='storageinsights.projects.locations.reportConfigs.create', ordered_params=['parent'], path_params=['parent'], query_params=['requestId'], relative_path='v1/{+parent}/reportConfigs', request_field='reportConfig', request_type_name='StorageinsightsProjectsLocationsReportConfigsCreateRequest', response_type_name='ReportConfig', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single ReportConfig.

      Args:
        request: (StorageinsightsProjectsLocationsReportConfigsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/reportConfigs/{reportConfigsId}', http_method='DELETE', method_id='storageinsights.projects.locations.reportConfigs.delete', ordered_params=['name'], path_params=['name'], query_params=['force', 'requestId'], relative_path='v1/{+name}', request_field='', request_type_name='StorageinsightsProjectsLocationsReportConfigsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single ReportConfig.

      Args:
        request: (StorageinsightsProjectsLocationsReportConfigsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ReportConfig) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/reportConfigs/{reportConfigsId}', http_method='GET', method_id='storageinsights.projects.locations.reportConfigs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='StorageinsightsProjectsLocationsReportConfigsGetRequest', response_type_name='ReportConfig', supports_download=False)

    def List(self, request, global_params=None):
        """Lists ReportConfigs in a given project and location.

      Args:
        request: (StorageinsightsProjectsLocationsReportConfigsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListReportConfigsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/reportConfigs', http_method='GET', method_id='storageinsights.projects.locations.reportConfigs.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/reportConfigs', request_field='', request_type_name='StorageinsightsProjectsLocationsReportConfigsListRequest', response_type_name='ListReportConfigsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single ReportConfig.

      Args:
        request: (StorageinsightsProjectsLocationsReportConfigsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ReportConfig) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/reportConfigs/{reportConfigsId}', http_method='PATCH', method_id='storageinsights.projects.locations.reportConfigs.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='reportConfig', request_type_name='StorageinsightsProjectsLocationsReportConfigsPatchRequest', response_type_name='ReportConfig', supports_download=False)
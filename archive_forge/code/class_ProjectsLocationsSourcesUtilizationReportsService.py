from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmmigration.v1 import vmmigration_v1_messages as messages
class ProjectsLocationsSourcesUtilizationReportsService(base_api.BaseApiService):
    """Service class for the projects_locations_sources_utilizationReports resource."""
    _NAME = 'projects_locations_sources_utilizationReports'

    def __init__(self, client):
        super(VmmigrationV1.ProjectsLocationsSourcesUtilizationReportsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new UtilizationReport.

      Args:
        request: (VmmigrationProjectsLocationsSourcesUtilizationReportsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sources/{sourcesId}/utilizationReports', http_method='POST', method_id='vmmigration.projects.locations.sources.utilizationReports.create', ordered_params=['parent'], path_params=['parent'], query_params=['requestId', 'utilizationReportId'], relative_path='v1/{+parent}/utilizationReports', request_field='utilizationReport', request_type_name='VmmigrationProjectsLocationsSourcesUtilizationReportsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single Utilization Report.

      Args:
        request: (VmmigrationProjectsLocationsSourcesUtilizationReportsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sources/{sourcesId}/utilizationReports/{utilizationReportsId}', http_method='DELETE', method_id='vmmigration.projects.locations.sources.utilizationReports.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='VmmigrationProjectsLocationsSourcesUtilizationReportsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a single Utilization Report.

      Args:
        request: (VmmigrationProjectsLocationsSourcesUtilizationReportsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (UtilizationReport) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sources/{sourcesId}/utilizationReports/{utilizationReportsId}', http_method='GET', method_id='vmmigration.projects.locations.sources.utilizationReports.get', ordered_params=['name'], path_params=['name'], query_params=['view'], relative_path='v1/{+name}', request_field='', request_type_name='VmmigrationProjectsLocationsSourcesUtilizationReportsGetRequest', response_type_name='UtilizationReport', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Utilization Reports of the given Source.

      Args:
        request: (VmmigrationProjectsLocationsSourcesUtilizationReportsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListUtilizationReportsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sources/{sourcesId}/utilizationReports', http_method='GET', method_id='vmmigration.projects.locations.sources.utilizationReports.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken', 'view'], relative_path='v1/{+parent}/utilizationReports', request_field='', request_type_name='VmmigrationProjectsLocationsSourcesUtilizationReportsListRequest', response_type_name='ListUtilizationReportsResponse', supports_download=False)
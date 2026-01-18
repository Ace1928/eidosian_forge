from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.run.v2 import run_v2_messages as messages
class ProjectsLocationsServicesRevisionsService(base_api.BaseApiService):
    """Service class for the projects_locations_services_revisions resource."""
    _NAME = 'projects_locations_services_revisions'

    def __init__(self, client):
        super(RunV2.ProjectsLocationsServicesRevisionsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes a Revision.

      Args:
        request: (RunProjectsLocationsServicesRevisionsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/services/{servicesId}/revisions/{revisionsId}', http_method='DELETE', method_id='run.projects.locations.services.revisions.delete', ordered_params=['name'], path_params=['name'], query_params=['etag', 'validateOnly'], relative_path='v2/{+name}', request_field='', request_type_name='RunProjectsLocationsServicesRevisionsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def ExportStatus(self, request, global_params=None):
        """Read the status of an image export operation.

      Args:
        request: (RunProjectsLocationsServicesRevisionsExportStatusRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRunV2ExportStatusResponse) The response message.
      """
        config = self.GetMethodConfig('ExportStatus')
        return self._RunMethod(config, request, global_params=global_params)
    ExportStatus.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/services/{servicesId}/revisions/{revisionsId}/{revisionsId1}:exportStatus', http_method='GET', method_id='run.projects.locations.services.revisions.exportStatus', ordered_params=['name', 'operationId'], path_params=['name', 'operationId'], query_params=[], relative_path='v2/{+name}/{+operationId}:exportStatus', request_field='', request_type_name='RunProjectsLocationsServicesRevisionsExportStatusRequest', response_type_name='GoogleCloudRunV2ExportStatusResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets information about a Revision.

      Args:
        request: (RunProjectsLocationsServicesRevisionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRunV2Revision) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/services/{servicesId}/revisions/{revisionsId}', http_method='GET', method_id='run.projects.locations.services.revisions.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='RunProjectsLocationsServicesRevisionsGetRequest', response_type_name='GoogleCloudRunV2Revision', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Revisions from a given Service, or from a given location.

      Args:
        request: (RunProjectsLocationsServicesRevisionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRunV2ListRevisionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/services/{servicesId}/revisions', http_method='GET', method_id='run.projects.locations.services.revisions.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'showDeleted'], relative_path='v2/{+parent}/revisions', request_field='', request_type_name='RunProjectsLocationsServicesRevisionsListRequest', response_type_name='GoogleCloudRunV2ListRevisionsResponse', supports_download=False)
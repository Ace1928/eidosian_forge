from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.metastore.v1beta import metastore_v1beta_messages as messages
class ProjectsLocationsServicesMetadataImportsService(base_api.BaseApiService):
    """Service class for the projects_locations_services_metadataImports resource."""
    _NAME = 'projects_locations_services_metadataImports'

    def __init__(self, client):
        super(MetastoreV1beta.ProjectsLocationsServicesMetadataImportsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new MetadataImport in a given project and location.

      Args:
        request: (MetastoreProjectsLocationsServicesMetadataImportsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/services/{servicesId}/metadataImports', http_method='POST', method_id='metastore.projects.locations.services.metadataImports.create', ordered_params=['parent'], path_params=['parent'], query_params=['metadataImportId', 'requestId'], relative_path='v1beta/{+parent}/metadataImports', request_field='metadataImport', request_type_name='MetastoreProjectsLocationsServicesMetadataImportsCreateRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single import.

      Args:
        request: (MetastoreProjectsLocationsServicesMetadataImportsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (MetadataImport) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/services/{servicesId}/metadataImports/{metadataImportsId}', http_method='GET', method_id='metastore.projects.locations.services.metadataImports.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='MetastoreProjectsLocationsServicesMetadataImportsGetRequest', response_type_name='MetadataImport', supports_download=False)

    def List(self, request, global_params=None):
        """Lists imports in a service.

      Args:
        request: (MetastoreProjectsLocationsServicesMetadataImportsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListMetadataImportsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/services/{servicesId}/metadataImports', http_method='GET', method_id='metastore.projects.locations.services.metadataImports.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1beta/{+parent}/metadataImports', request_field='', request_type_name='MetastoreProjectsLocationsServicesMetadataImportsListRequest', response_type_name='ListMetadataImportsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a single import. Only the description field of MetadataImport is supported to be updated.

      Args:
        request: (MetastoreProjectsLocationsServicesMetadataImportsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/services/{servicesId}/metadataImports/{metadataImportsId}', http_method='PATCH', method_id='metastore.projects.locations.services.metadataImports.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1beta/{+name}', request_field='metadataImport', request_type_name='MetastoreProjectsLocationsServicesMetadataImportsPatchRequest', response_type_name='Operation', supports_download=False)
from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.mediaasset.v1alpha import mediaasset_v1alpha_messages as messages
class ProjectsLocationsCatalogsService(base_api.BaseApiService):
    """Service class for the projects_locations_catalogs resource."""
    _NAME = 'projects_locations_catalogs'

    def __init__(self, client):
        super(MediaassetV1alpha.ProjectsLocationsCatalogsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new catalog in a given project and location.

      Args:
        request: (MediaassetProjectsLocationsCatalogsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/catalogs', http_method='POST', method_id='mediaasset.projects.locations.catalogs.create', ordered_params=['parent'], path_params=['parent'], query_params=['catalogId'], relative_path='v1alpha/{+parent}/catalogs', request_field='catalog', request_type_name='MediaassetProjectsLocationsCatalogsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single catalog.

      Args:
        request: (MediaassetProjectsLocationsCatalogsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/catalogs/{catalogsId}', http_method='DELETE', method_id='mediaasset.projects.locations.catalogs.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='MediaassetProjectsLocationsCatalogsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single catalog.

      Args:
        request: (MediaassetProjectsLocationsCatalogsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Catalog) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/catalogs/{catalogsId}', http_method='GET', method_id='mediaasset.projects.locations.catalogs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='MediaassetProjectsLocationsCatalogsGetRequest', response_type_name='Catalog', supports_download=False)

    def List(self, request, global_params=None):
        """Lists catalogs in a given project and location.

      Args:
        request: (MediaassetProjectsLocationsCatalogsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListCatalogsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/catalogs', http_method='GET', method_id='mediaasset.projects.locations.catalogs.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/catalogs', request_field='', request_type_name='MediaassetProjectsLocationsCatalogsListRequest', response_type_name='ListCatalogsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single catalog.

      Args:
        request: (MediaassetProjectsLocationsCatalogsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/catalogs/{catalogsId}', http_method='PATCH', method_id='mediaasset.projects.locations.catalogs.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha/{+name}', request_field='catalog', request_type_name='MediaassetProjectsLocationsCatalogsPatchRequest', response_type_name='Operation', supports_download=False)

    def Search(self, request, global_params=None):
        """Search returns the resources (e.g., assets and annotations) under a Catalog that match the given query. Search covers both media content and metadata.

      Args:
        request: (MediaassetProjectsLocationsCatalogsSearchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CatalogSearchResponse) The response message.
      """
        config = self.GetMethodConfig('Search')
        return self._RunMethod(config, request, global_params=global_params)
    Search.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/catalogs/{catalogsId}:search', http_method='POST', method_id='mediaasset.projects.locations.catalogs.search', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}:search', request_field='catalogSearchRequest', request_type_name='MediaassetProjectsLocationsCatalogsSearchRequest', response_type_name='CatalogSearchResponse', supports_download=False)
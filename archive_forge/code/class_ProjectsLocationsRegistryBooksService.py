from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudnumberregistry.v1alpha import cloudnumberregistry_v1alpha_messages as messages
class ProjectsLocationsRegistryBooksService(base_api.BaseApiService):
    """Service class for the projects_locations_registryBooks resource."""
    _NAME = 'projects_locations_registryBooks'

    def __init__(self, client):
        super(CloudnumberregistryV1alpha.ProjectsLocationsRegistryBooksService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new RegistryBook in a given project and location.

      Args:
        request: (CloudnumberregistryProjectsLocationsRegistryBooksCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/registryBooks', http_method='POST', method_id='cloudnumberregistry.projects.locations.registryBooks.create', ordered_params=['parent'], path_params=['parent'], query_params=['registryBookId', 'requestId'], relative_path='v1alpha/{+parent}/registryBooks', request_field='registryBook', request_type_name='CloudnumberregistryProjectsLocationsRegistryBooksCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single RegistryBook.

      Args:
        request: (CloudnumberregistryProjectsLocationsRegistryBooksDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/registryBooks/{registryBooksId}', http_method='DELETE', method_id='cloudnumberregistry.projects.locations.registryBooks.delete', ordered_params=['name'], path_params=['name'], query_params=['force', 'requestId'], relative_path='v1alpha/{+name}', request_field='', request_type_name='CloudnumberregistryProjectsLocationsRegistryBooksDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single RegistryBook.

      Args:
        request: (CloudnumberregistryProjectsLocationsRegistryBooksGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RegistryBook) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/registryBooks/{registryBooksId}', http_method='GET', method_id='cloudnumberregistry.projects.locations.registryBooks.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='CloudnumberregistryProjectsLocationsRegistryBooksGetRequest', response_type_name='RegistryBook', supports_download=False)

    def HistoricalEvents(self, request, global_params=None):
        """Shows HistoricalEvents in a given registry book.

      Args:
        request: (CloudnumberregistryProjectsLocationsRegistryBooksHistoricalEventsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ShowHistoricalEventsResponse) The response message.
      """
        config = self.GetMethodConfig('HistoricalEvents')
        return self._RunMethod(config, request, global_params=global_params)
    HistoricalEvents.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/registryBooks/{registryBooksId}/historicalEvents', http_method='GET', method_id='cloudnumberregistry.projects.locations.registryBooks.historicalEvents', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/historicalEvents', request_field='', request_type_name='CloudnumberregistryProjectsLocationsRegistryBooksHistoricalEventsRequest', response_type_name='ShowHistoricalEventsResponse', supports_download=False)

    def List(self, request, global_params=None):
        """Lists RegistryBooks in a given project and location.

      Args:
        request: (CloudnumberregistryProjectsLocationsRegistryBooksListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListRegistryBooksResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/registryBooks', http_method='GET', method_id='cloudnumberregistry.projects.locations.registryBooks.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/registryBooks', request_field='', request_type_name='CloudnumberregistryProjectsLocationsRegistryBooksListRequest', response_type_name='ListRegistryBooksResponse', supports_download=False)

    def NodeEvents(self, request, global_params=None):
        """Shows NodeEvents related to an IP range in a given registry book.

      Args:
        request: (CloudnumberregistryProjectsLocationsRegistryBooksNodeEventsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ShowNodeEventsResponse) The response message.
      """
        config = self.GetMethodConfig('NodeEvents')
        return self._RunMethod(config, request, global_params=global_params)
    NodeEvents.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/registryBooks/{registryBooksId}/nodeEvents', http_method='GET', method_id='cloudnumberregistry.projects.locations.registryBooks.nodeEvents', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'ipRange', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/nodeEvents', request_field='', request_type_name='CloudnumberregistryProjectsLocationsRegistryBooksNodeEventsRequest', response_type_name='ShowNodeEventsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single RegistryBook.

      Args:
        request: (CloudnumberregistryProjectsLocationsRegistryBooksPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/registryBooks/{registryBooksId}', http_method='PATCH', method_id='cloudnumberregistry.projects.locations.registryBooks.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1alpha/{+name}', request_field='registryBook', request_type_name='CloudnumberregistryProjectsLocationsRegistryBooksPatchRequest', response_type_name='Operation', supports_download=False)

    def SearchRegistry(self, request, global_params=None):
        """Search registry nodes in a given registry book.

      Args:
        request: (CloudnumberregistryProjectsLocationsRegistryBooksSearchRegistryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SearchRegistryResponse) The response message.
      """
        config = self.GetMethodConfig('SearchRegistry')
        return self._RunMethod(config, request, global_params=global_params)
    SearchRegistry.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/registryBooks/{registryBooksId}:searchRegistry', http_method='GET', method_id='cloudnumberregistry.projects.locations.registryBooks.searchRegistry', ordered_params=['book'], path_params=['book'], query_params=['attributeKeys', 'ipRange', 'keywords', 'orderBy', 'pageSize', 'pageToken', 'source'], relative_path='v1alpha/{+book}:searchRegistry', request_field='', request_type_name='CloudnumberregistryProjectsLocationsRegistryBooksSearchRegistryRequest', response_type_name='SearchRegistryResponse', supports_download=False)
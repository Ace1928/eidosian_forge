from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networkconnectivity.v1 import networkconnectivity_v1_messages as messages
class ProjectsLocationsServiceConnectionTokensService(base_api.BaseApiService):
    """Service class for the projects_locations_serviceConnectionTokens resource."""
    _NAME = 'projects_locations_serviceConnectionTokens'

    def __init__(self, client):
        super(NetworkconnectivityV1.ProjectsLocationsServiceConnectionTokensService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new ServiceConnectionToken in a given project and location.

      Args:
        request: (NetworkconnectivityProjectsLocationsServiceConnectionTokensCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/serviceConnectionTokens', http_method='POST', method_id='networkconnectivity.projects.locations.serviceConnectionTokens.create', ordered_params=['parent'], path_params=['parent'], query_params=['requestId', 'serviceConnectionTokenId'], relative_path='v1/{+parent}/serviceConnectionTokens', request_field='serviceConnectionToken', request_type_name='NetworkconnectivityProjectsLocationsServiceConnectionTokensCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single ServiceConnectionToken.

      Args:
        request: (NetworkconnectivityProjectsLocationsServiceConnectionTokensDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/serviceConnectionTokens/{serviceConnectionTokensId}', http_method='DELETE', method_id='networkconnectivity.projects.locations.serviceConnectionTokens.delete', ordered_params=['name'], path_params=['name'], query_params=['etag', 'requestId'], relative_path='v1/{+name}', request_field='', request_type_name='NetworkconnectivityProjectsLocationsServiceConnectionTokensDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single ServiceConnectionToken.

      Args:
        request: (NetworkconnectivityProjectsLocationsServiceConnectionTokensGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ServiceConnectionToken) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/serviceConnectionTokens/{serviceConnectionTokensId}', http_method='GET', method_id='networkconnectivity.projects.locations.serviceConnectionTokens.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworkconnectivityProjectsLocationsServiceConnectionTokensGetRequest', response_type_name='ServiceConnectionToken', supports_download=False)

    def List(self, request, global_params=None):
        """Lists ServiceConnectionTokens in a given project and location.

      Args:
        request: (NetworkconnectivityProjectsLocationsServiceConnectionTokensListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListServiceConnectionTokensResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/serviceConnectionTokens', http_method='GET', method_id='networkconnectivity.projects.locations.serviceConnectionTokens.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/serviceConnectionTokens', request_field='', request_type_name='NetworkconnectivityProjectsLocationsServiceConnectionTokensListRequest', response_type_name='ListServiceConnectionTokensResponse', supports_download=False)
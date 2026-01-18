from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networkconnectivity.v1beta import networkconnectivity_v1beta_messages as messages
class ProjectsLocationsRegionalEndpointsService(base_api.BaseApiService):
    """Service class for the projects_locations_regionalEndpoints resource."""
    _NAME = 'projects_locations_regionalEndpoints'

    def __init__(self, client):
        super(NetworkconnectivityV1beta.ProjectsLocationsRegionalEndpointsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new RegionalEndpoint in a given project and location.

      Args:
        request: (NetworkconnectivityProjectsLocationsRegionalEndpointsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/regionalEndpoints', http_method='POST', method_id='networkconnectivity.projects.locations.regionalEndpoints.create', ordered_params=['parent'], path_params=['parent'], query_params=['regionalEndpointId', 'requestId'], relative_path='v1beta/{+parent}/regionalEndpoints', request_field='googleCloudNetworkconnectivityV1betaRegionalEndpoint', request_type_name='NetworkconnectivityProjectsLocationsRegionalEndpointsCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single RegionalEndpoint.

      Args:
        request: (NetworkconnectivityProjectsLocationsRegionalEndpointsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/regionalEndpoints/{regionalEndpointsId}', http_method='DELETE', method_id='networkconnectivity.projects.locations.regionalEndpoints.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1beta/{+name}', request_field='', request_type_name='NetworkconnectivityProjectsLocationsRegionalEndpointsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single RegionalEndpoint.

      Args:
        request: (NetworkconnectivityProjectsLocationsRegionalEndpointsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudNetworkconnectivityV1betaRegionalEndpoint) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/regionalEndpoints/{regionalEndpointsId}', http_method='GET', method_id='networkconnectivity.projects.locations.regionalEndpoints.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='NetworkconnectivityProjectsLocationsRegionalEndpointsGetRequest', response_type_name='GoogleCloudNetworkconnectivityV1betaRegionalEndpoint', supports_download=False)

    def List(self, request, global_params=None):
        """Lists RegionalEndpoints in a given project and location.

      Args:
        request: (NetworkconnectivityProjectsLocationsRegionalEndpointsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudNetworkconnectivityV1betaListRegionalEndpointsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/regionalEndpoints', http_method='GET', method_id='networkconnectivity.projects.locations.regionalEndpoints.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1beta/{+parent}/regionalEndpoints', request_field='', request_type_name='NetworkconnectivityProjectsLocationsRegionalEndpointsListRequest', response_type_name='GoogleCloudNetworkconnectivityV1betaListRegionalEndpointsResponse', supports_download=False)
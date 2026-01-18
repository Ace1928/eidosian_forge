from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.baremetalsolution.v2 import baremetalsolution_v2_messages as messages
class ProjectsLocationsNetworksService(base_api.BaseApiService):
    """Service class for the projects_locations_networks resource."""
    _NAME = 'projects_locations_networks'

    def __init__(self, client):
        super(BaremetalsolutionV2.ProjectsLocationsNetworksService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Get details of a single network.

      Args:
        request: (BaremetalsolutionProjectsLocationsNetworksGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Network) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/networks/{networksId}', http_method='GET', method_id='baremetalsolution.projects.locations.networks.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='BaremetalsolutionProjectsLocationsNetworksGetRequest', response_type_name='Network', supports_download=False)

    def List(self, request, global_params=None):
        """List network in a given project and location.

      Args:
        request: (BaremetalsolutionProjectsLocationsNetworksListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListNetworksResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/networks', http_method='GET', method_id='baremetalsolution.projects.locations.networks.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v2/{+parent}/networks', request_field='', request_type_name='BaremetalsolutionProjectsLocationsNetworksListRequest', response_type_name='ListNetworksResponse', supports_download=False)

    def ListNetworkUsage(self, request, global_params=None):
        """List all Networks (and used IPs for each Network) in the vendor account associated with the specified project.

      Args:
        request: (BaremetalsolutionProjectsLocationsNetworksListNetworkUsageRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListNetworkUsageResponse) The response message.
      """
        config = self.GetMethodConfig('ListNetworkUsage')
        return self._RunMethod(config, request, global_params=global_params)
    ListNetworkUsage.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/networks:listNetworkUsage', http_method='GET', method_id='baremetalsolution.projects.locations.networks.listNetworkUsage', ordered_params=['location'], path_params=['location'], query_params=[], relative_path='v2/{+location}/networks:listNetworkUsage', request_field='', request_type_name='BaremetalsolutionProjectsLocationsNetworksListNetworkUsageRequest', response_type_name='ListNetworkUsageResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Update details of a single network.

      Args:
        request: (BaremetalsolutionProjectsLocationsNetworksPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/networks/{networksId}', http_method='PATCH', method_id='baremetalsolution.projects.locations.networks.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}', request_field='network', request_type_name='BaremetalsolutionProjectsLocationsNetworksPatchRequest', response_type_name='Operation', supports_download=False)

    def Rename(self, request, global_params=None):
        """RenameNetwork sets a new name for a network. Use with caution, previous names become immediately invalidated.

      Args:
        request: (BaremetalsolutionProjectsLocationsNetworksRenameRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Network) The response message.
      """
        config = self.GetMethodConfig('Rename')
        return self._RunMethod(config, request, global_params=global_params)
    Rename.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/networks/{networksId}:rename', http_method='POST', method_id='baremetalsolution.projects.locations.networks.rename', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:rename', request_field='renameNetworkRequest', request_type_name='BaremetalsolutionProjectsLocationsNetworksRenameRequest', response_type_name='Network', supports_download=False)
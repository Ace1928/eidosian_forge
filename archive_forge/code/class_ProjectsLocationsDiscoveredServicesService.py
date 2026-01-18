from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apphub.v1alpha import apphub_v1alpha_messages as messages
class ProjectsLocationsDiscoveredServicesService(base_api.BaseApiService):
    """Service class for the projects_locations_discoveredServices resource."""
    _NAME = 'projects_locations_discoveredServices'

    def __init__(self, client):
        super(ApphubV1alpha.ProjectsLocationsDiscoveredServicesService, self).__init__(client)
        self._upload_configs = {}

    def FindUnregistered(self, request, global_params=None):
        """Finds unregistered services in a host project and location.

      Args:
        request: (ApphubProjectsLocationsDiscoveredServicesFindUnregisteredRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FindUnregisteredServicesResponse) The response message.
      """
        config = self.GetMethodConfig('FindUnregistered')
        return self._RunMethod(config, request, global_params=global_params)
    FindUnregistered.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/discoveredServices:findUnregistered', http_method='GET', method_id='apphub.projects.locations.discoveredServices.findUnregistered', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/discoveredServices:findUnregistered', request_field='', request_type_name='ApphubProjectsLocationsDiscoveredServicesFindUnregisteredRequest', response_type_name='FindUnregisteredServicesResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a Discovered Service in a host project and location.

      Args:
        request: (ApphubProjectsLocationsDiscoveredServicesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DiscoveredService) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/discoveredServices/{discoveredServicesId}', http_method='GET', method_id='apphub.projects.locations.discoveredServices.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='ApphubProjectsLocationsDiscoveredServicesGetRequest', response_type_name='DiscoveredService', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Discovered Services that can be added to an Application in a host project and location.

      Args:
        request: (ApphubProjectsLocationsDiscoveredServicesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDiscoveredServicesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/discoveredServices', http_method='GET', method_id='apphub.projects.locations.discoveredServices.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/discoveredServices', request_field='', request_type_name='ApphubProjectsLocationsDiscoveredServicesListRequest', response_type_name='ListDiscoveredServicesResponse', supports_download=False)

    def Lookup(self, request, global_params=None):
        """Lists a Discovered Service in a host project and location, with a given resource URI.

      Args:
        request: (ApphubProjectsLocationsDiscoveredServicesLookupRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LookupDiscoveredServiceResponse) The response message.
      """
        config = self.GetMethodConfig('Lookup')
        return self._RunMethod(config, request, global_params=global_params)
    Lookup.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/discoveredServices:lookup', http_method='GET', method_id='apphub.projects.locations.discoveredServices.lookup', ordered_params=['parent'], path_params=['parent'], query_params=['uri'], relative_path='v1alpha/{+parent}/discoveredServices:lookup', request_field='', request_type_name='ApphubProjectsLocationsDiscoveredServicesLookupRequest', response_type_name='LookupDiscoveredServiceResponse', supports_download=False)
from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apphub.v1alpha import apphub_v1alpha_messages as messages
class ProjectsLocationsDiscoveredWorkloadsService(base_api.BaseApiService):
    """Service class for the projects_locations_discoveredWorkloads resource."""
    _NAME = 'projects_locations_discoveredWorkloads'

    def __init__(self, client):
        super(ApphubV1alpha.ProjectsLocationsDiscoveredWorkloadsService, self).__init__(client)
        self._upload_configs = {}

    def FindUnregistered(self, request, global_params=None):
        """Finds unregistered workloads in a host project and location.

      Args:
        request: (ApphubProjectsLocationsDiscoveredWorkloadsFindUnregisteredRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FindUnregisteredWorkloadsResponse) The response message.
      """
        config = self.GetMethodConfig('FindUnregistered')
        return self._RunMethod(config, request, global_params=global_params)
    FindUnregistered.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/discoveredWorkloads:findUnregistered', http_method='GET', method_id='apphub.projects.locations.discoveredWorkloads.findUnregistered', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/discoveredWorkloads:findUnregistered', request_field='', request_type_name='ApphubProjectsLocationsDiscoveredWorkloadsFindUnregisteredRequest', response_type_name='FindUnregisteredWorkloadsResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a Discovered Workload in a host project and location.

      Args:
        request: (ApphubProjectsLocationsDiscoveredWorkloadsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DiscoveredWorkload) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/discoveredWorkloads/{discoveredWorkloadsId}', http_method='GET', method_id='apphub.projects.locations.discoveredWorkloads.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='ApphubProjectsLocationsDiscoveredWorkloadsGetRequest', response_type_name='DiscoveredWorkload', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Discovered Workloads that can be added to an Application in a host project and location.

      Args:
        request: (ApphubProjectsLocationsDiscoveredWorkloadsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDiscoveredWorkloadsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/discoveredWorkloads', http_method='GET', method_id='apphub.projects.locations.discoveredWorkloads.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/discoveredWorkloads', request_field='', request_type_name='ApphubProjectsLocationsDiscoveredWorkloadsListRequest', response_type_name='ListDiscoveredWorkloadsResponse', supports_download=False)

    def Lookup(self, request, global_params=None):
        """Lists a Discovered Workload in a host project and location, with a given resource URI.

      Args:
        request: (ApphubProjectsLocationsDiscoveredWorkloadsLookupRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LookupDiscoveredWorkloadResponse) The response message.
      """
        config = self.GetMethodConfig('Lookup')
        return self._RunMethod(config, request, global_params=global_params)
    Lookup.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/discoveredWorkloads:lookup', http_method='GET', method_id='apphub.projects.locations.discoveredWorkloads.lookup', ordered_params=['parent'], path_params=['parent'], query_params=['uri'], relative_path='v1alpha/{+parent}/discoveredWorkloads:lookup', request_field='', request_type_name='ApphubProjectsLocationsDiscoveredWorkloadsLookupRequest', response_type_name='LookupDiscoveredWorkloadResponse', supports_download=False)
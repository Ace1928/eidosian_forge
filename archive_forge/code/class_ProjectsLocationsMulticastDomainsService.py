from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networkservices.v1 import networkservices_v1_messages as messages
class ProjectsLocationsMulticastDomainsService(base_api.BaseApiService):
    """Service class for the projects_locations_multicastDomains resource."""
    _NAME = 'projects_locations_multicastDomains'

    def __init__(self, client):
        super(NetworkservicesV1.ProjectsLocationsMulticastDomainsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new multicast domain in a given project and location.

      Args:
        request: (NetworkservicesProjectsLocationsMulticastDomainsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/multicastDomains', http_method='POST', method_id='networkservices.projects.locations.multicastDomains.create', ordered_params=['parent'], path_params=['parent'], query_params=['multicastDomainId', 'requestId'], relative_path='v1/{+parent}/multicastDomains', request_field='multicastDomain', request_type_name='NetworkservicesProjectsLocationsMulticastDomainsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single multicast domain.

      Args:
        request: (NetworkservicesProjectsLocationsMulticastDomainsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/multicastDomains/{multicastDomainsId}', http_method='DELETE', method_id='networkservices.projects.locations.multicastDomains.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='NetworkservicesProjectsLocationsMulticastDomainsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single multicast domain.

      Args:
        request: (NetworkservicesProjectsLocationsMulticastDomainsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (MulticastDomain) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/multicastDomains/{multicastDomainsId}', http_method='GET', method_id='networkservices.projects.locations.multicastDomains.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworkservicesProjectsLocationsMulticastDomainsGetRequest', response_type_name='MulticastDomain', supports_download=False)

    def List(self, request, global_params=None):
        """Lists multicast domains in a given project and location.

      Args:
        request: (NetworkservicesProjectsLocationsMulticastDomainsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListMulticastDomainsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/multicastDomains', http_method='GET', method_id='networkservices.projects.locations.multicastDomains.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/multicastDomains', request_field='', request_type_name='NetworkservicesProjectsLocationsMulticastDomainsListRequest', response_type_name='ListMulticastDomainsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single multicast domain.

      Args:
        request: (NetworkservicesProjectsLocationsMulticastDomainsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/multicastDomains/{multicastDomainsId}', http_method='PATCH', method_id='networkservices.projects.locations.multicastDomains.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='multicastDomain', request_type_name='NetworkservicesProjectsLocationsMulticastDomainsPatchRequest', response_type_name='Operation', supports_download=False)
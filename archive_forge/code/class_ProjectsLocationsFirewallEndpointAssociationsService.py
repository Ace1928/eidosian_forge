from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.networksecurity.v1 import networksecurity_v1_messages as messages
class ProjectsLocationsFirewallEndpointAssociationsService(base_api.BaseApiService):
    """Service class for the projects_locations_firewallEndpointAssociations resource."""
    _NAME = 'projects_locations_firewallEndpointAssociations'

    def __init__(self, client):
        super(NetworksecurityV1.ProjectsLocationsFirewallEndpointAssociationsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new FirewallEndpointAssociation in a given project and location.

      Args:
        request: (NetworksecurityProjectsLocationsFirewallEndpointAssociationsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/firewallEndpointAssociations', http_method='POST', method_id='networksecurity.projects.locations.firewallEndpointAssociations.create', ordered_params=['parent'], path_params=['parent'], query_params=['firewallEndpointAssociationId', 'requestId'], relative_path='v1/{+parent}/firewallEndpointAssociations', request_field='firewallEndpointAssociation', request_type_name='NetworksecurityProjectsLocationsFirewallEndpointAssociationsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single FirewallEndpointAssociation.

      Args:
        request: (NetworksecurityProjectsLocationsFirewallEndpointAssociationsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/firewallEndpointAssociations/{firewallEndpointAssociationsId}', http_method='DELETE', method_id='networksecurity.projects.locations.firewallEndpointAssociations.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='NetworksecurityProjectsLocationsFirewallEndpointAssociationsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single FirewallEndpointAssociation.

      Args:
        request: (NetworksecurityProjectsLocationsFirewallEndpointAssociationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FirewallEndpointAssociation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/firewallEndpointAssociations/{firewallEndpointAssociationsId}', http_method='GET', method_id='networksecurity.projects.locations.firewallEndpointAssociations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='NetworksecurityProjectsLocationsFirewallEndpointAssociationsGetRequest', response_type_name='FirewallEndpointAssociation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Associations in a given project and location.

      Args:
        request: (NetworksecurityProjectsLocationsFirewallEndpointAssociationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListFirewallEndpointAssociationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/firewallEndpointAssociations', http_method='GET', method_id='networksecurity.projects.locations.firewallEndpointAssociations.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/firewallEndpointAssociations', request_field='', request_type_name='NetworksecurityProjectsLocationsFirewallEndpointAssociationsListRequest', response_type_name='ListFirewallEndpointAssociationsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Update a single FirewallEndpointAssociation.

      Args:
        request: (NetworksecurityProjectsLocationsFirewallEndpointAssociationsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/firewallEndpointAssociations/{firewallEndpointAssociationsId}', http_method='PATCH', method_id='networksecurity.projects.locations.firewallEndpointAssociations.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='firewallEndpointAssociation', request_type_name='NetworksecurityProjectsLocationsFirewallEndpointAssociationsPatchRequest', response_type_name='Operation', supports_download=False)
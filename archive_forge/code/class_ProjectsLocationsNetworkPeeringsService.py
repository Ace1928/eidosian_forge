from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmwareengine.v1 import vmwareengine_v1_messages as messages
class ProjectsLocationsNetworkPeeringsService(base_api.BaseApiService):
    """Service class for the projects_locations_networkPeerings resource."""
    _NAME = 'projects_locations_networkPeerings'

    def __init__(self, client):
        super(VmwareengineV1.ProjectsLocationsNetworkPeeringsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new network peering between the peer network and VMware Engine network provided in a `NetworkPeering` resource. NetworkPeering is a global resource and location can only be global.

      Args:
        request: (VmwareengineProjectsLocationsNetworkPeeringsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/networkPeerings', http_method='POST', method_id='vmwareengine.projects.locations.networkPeerings.create', ordered_params=['parent'], path_params=['parent'], query_params=['networkPeeringId', 'requestId'], relative_path='v1/{+parent}/networkPeerings', request_field='networkPeering', request_type_name='VmwareengineProjectsLocationsNetworkPeeringsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a `NetworkPeering` resource. When a network peering is deleted for a VMware Engine network, the peer network becomes inaccessible to that VMware Engine network. NetworkPeering is a global resource and location can only be global.

      Args:
        request: (VmwareengineProjectsLocationsNetworkPeeringsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/networkPeerings/{networkPeeringsId}', http_method='DELETE', method_id='vmwareengine.projects.locations.networkPeerings.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1/{+name}', request_field='', request_type_name='VmwareengineProjectsLocationsNetworkPeeringsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves a `NetworkPeering` resource by its resource name. The resource contains details of the network peering, such as peered networks, import and export custom route configurations, and peering state. NetworkPeering is a global resource and location can only be global.

      Args:
        request: (VmwareengineProjectsLocationsNetworkPeeringsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (NetworkPeering) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/networkPeerings/{networkPeeringsId}', http_method='GET', method_id='vmwareengine.projects.locations.networkPeerings.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='VmwareengineProjectsLocationsNetworkPeeringsGetRequest', response_type_name='NetworkPeering', supports_download=False)

    def List(self, request, global_params=None):
        """Lists `NetworkPeering` resources in a given project. NetworkPeering is a global resource and location can only be global.

      Args:
        request: (VmwareengineProjectsLocationsNetworkPeeringsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListNetworkPeeringsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/networkPeerings', http_method='GET', method_id='vmwareengine.projects.locations.networkPeerings.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/networkPeerings', request_field='', request_type_name='VmwareengineProjectsLocationsNetworkPeeringsListRequest', response_type_name='ListNetworkPeeringsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Modifies a `NetworkPeering` resource. Only the `description` field can be updated. Only fields specified in `updateMask` are applied. NetworkPeering is a global resource and location can only be global.

      Args:
        request: (VmwareengineProjectsLocationsNetworkPeeringsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/networkPeerings/{networkPeeringsId}', http_method='PATCH', method_id='vmwareengine.projects.locations.networkPeerings.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='networkPeering', request_type_name='VmwareengineProjectsLocationsNetworkPeeringsPatchRequest', response_type_name='Operation', supports_download=False)
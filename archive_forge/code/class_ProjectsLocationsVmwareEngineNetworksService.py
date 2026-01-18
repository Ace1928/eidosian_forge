from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmwareengine.v1 import vmwareengine_v1_messages as messages
class ProjectsLocationsVmwareEngineNetworksService(base_api.BaseApiService):
    """Service class for the projects_locations_vmwareEngineNetworks resource."""
    _NAME = 'projects_locations_vmwareEngineNetworks'

    def __init__(self, client):
        super(VmwareengineV1.ProjectsLocationsVmwareEngineNetworksService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new VMware Engine network that can be used by a private cloud.

      Args:
        request: (VmwareengineProjectsLocationsVmwareEngineNetworksCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/vmwareEngineNetworks', http_method='POST', method_id='vmwareengine.projects.locations.vmwareEngineNetworks.create', ordered_params=['parent'], path_params=['parent'], query_params=['requestId', 'vmwareEngineNetworkId'], relative_path='v1/{+parent}/vmwareEngineNetworks', request_field='vmwareEngineNetwork', request_type_name='VmwareengineProjectsLocationsVmwareEngineNetworksCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a `VmwareEngineNetwork` resource. You can only delete a VMware Engine network after all resources that refer to it are deleted. For example, a private cloud, a network peering, and a network policy can all refer to the same VMware Engine network.

      Args:
        request: (VmwareengineProjectsLocationsVmwareEngineNetworksDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/vmwareEngineNetworks/{vmwareEngineNetworksId}', http_method='DELETE', method_id='vmwareengine.projects.locations.vmwareEngineNetworks.delete', ordered_params=['name'], path_params=['name'], query_params=['etag', 'requestId'], relative_path='v1/{+name}', request_field='', request_type_name='VmwareengineProjectsLocationsVmwareEngineNetworksDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves a `VmwareEngineNetwork` resource by its resource name. The resource contains details of the VMware Engine network, such as its VMware Engine network type, peered networks in a service project, and state (for example, `CREATING`, `ACTIVE`, `DELETING`).

      Args:
        request: (VmwareengineProjectsLocationsVmwareEngineNetworksGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (VmwareEngineNetwork) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/vmwareEngineNetworks/{vmwareEngineNetworksId}', http_method='GET', method_id='vmwareengine.projects.locations.vmwareEngineNetworks.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='VmwareengineProjectsLocationsVmwareEngineNetworksGetRequest', response_type_name='VmwareEngineNetwork', supports_download=False)

    def List(self, request, global_params=None):
        """Lists `VmwareEngineNetwork` resources in a given project and location.

      Args:
        request: (VmwareengineProjectsLocationsVmwareEngineNetworksListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListVmwareEngineNetworksResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/vmwareEngineNetworks', http_method='GET', method_id='vmwareengine.projects.locations.vmwareEngineNetworks.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/vmwareEngineNetworks', request_field='', request_type_name='VmwareengineProjectsLocationsVmwareEngineNetworksListRequest', response_type_name='ListVmwareEngineNetworksResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Modifies a VMware Engine network resource. Only the following fields can be updated: `description`. Only fields specified in `updateMask` are applied.

      Args:
        request: (VmwareengineProjectsLocationsVmwareEngineNetworksPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/vmwareEngineNetworks/{vmwareEngineNetworksId}', http_method='PATCH', method_id='vmwareengine.projects.locations.vmwareEngineNetworks.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='vmwareEngineNetwork', request_type_name='VmwareengineProjectsLocationsVmwareEngineNetworksPatchRequest', response_type_name='Operation', supports_download=False)
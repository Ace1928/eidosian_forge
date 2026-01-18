from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.tpu.v2 import tpu_v2_messages as messages
class ProjectsLocationsNodesService(base_api.BaseApiService):
    """Service class for the projects_locations_nodes resource."""
    _NAME = 'projects_locations_nodes'

    def __init__(self, client):
        super(TpuV2.ProjectsLocationsNodesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a node.

      Args:
        request: (TpuProjectsLocationsNodesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/nodes', http_method='POST', method_id='tpu.projects.locations.nodes.create', ordered_params=['parent'], path_params=['parent'], query_params=['nodeId'], relative_path='v2/{+parent}/nodes', request_field='node', request_type_name='TpuProjectsLocationsNodesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a node.

      Args:
        request: (TpuProjectsLocationsNodesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/nodes/{nodesId}', http_method='DELETE', method_id='tpu.projects.locations.nodes.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='TpuProjectsLocationsNodesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the details of a node.

      Args:
        request: (TpuProjectsLocationsNodesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Node) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/nodes/{nodesId}', http_method='GET', method_id='tpu.projects.locations.nodes.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='TpuProjectsLocationsNodesGetRequest', response_type_name='Node', supports_download=False)

    def GetGuestAttributes(self, request, global_params=None):
        """Retrieves the guest attributes for the node.

      Args:
        request: (TpuProjectsLocationsNodesGetGuestAttributesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GetGuestAttributesResponse) The response message.
      """
        config = self.GetMethodConfig('GetGuestAttributes')
        return self._RunMethod(config, request, global_params=global_params)
    GetGuestAttributes.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/nodes/{nodesId}:getGuestAttributes', http_method='POST', method_id='tpu.projects.locations.nodes.getGuestAttributes', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:getGuestAttributes', request_field='getGuestAttributesRequest', request_type_name='TpuProjectsLocationsNodesGetGuestAttributesRequest', response_type_name='GetGuestAttributesResponse', supports_download=False)

    def List(self, request, global_params=None):
        """Lists nodes.

      Args:
        request: (TpuProjectsLocationsNodesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListNodesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/nodes', http_method='GET', method_id='tpu.projects.locations.nodes.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/nodes', request_field='', request_type_name='TpuProjectsLocationsNodesListRequest', response_type_name='ListNodesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the configurations of a node.

      Args:
        request: (TpuProjectsLocationsNodesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/nodes/{nodesId}', http_method='PATCH', method_id='tpu.projects.locations.nodes.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}', request_field='node', request_type_name='TpuProjectsLocationsNodesPatchRequest', response_type_name='Operation', supports_download=False)

    def Start(self, request, global_params=None):
        """Starts a node.

      Args:
        request: (TpuProjectsLocationsNodesStartRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Start')
        return self._RunMethod(config, request, global_params=global_params)
    Start.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/nodes/{nodesId}:start', http_method='POST', method_id='tpu.projects.locations.nodes.start', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:start', request_field='startNodeRequest', request_type_name='TpuProjectsLocationsNodesStartRequest', response_type_name='Operation', supports_download=False)

    def Stop(self, request, global_params=None):
        """Stops a node. This operation is only available with single TPU nodes.

      Args:
        request: (TpuProjectsLocationsNodesStopRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Stop')
        return self._RunMethod(config, request, global_params=global_params)
    Stop.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/nodes/{nodesId}:stop', http_method='POST', method_id='tpu.projects.locations.nodes.stop', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:stop', request_field='stopNodeRequest', request_type_name='TpuProjectsLocationsNodesStopRequest', response_type_name='Operation', supports_download=False)
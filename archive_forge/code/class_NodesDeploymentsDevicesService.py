from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sasportal.v1alpha1 import sasportal_v1alpha1_messages as messages
class NodesDeploymentsDevicesService(base_api.BaseApiService):
    """Service class for the nodes_deployments_devices resource."""
    _NAME = 'nodes_deployments_devices'

    def __init__(self, client):
        super(SasportalV1alpha1.NodesDeploymentsDevicesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a device under a node or customer.

      Args:
        request: (SasportalNodesDeploymentsDevicesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalDevice) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/nodes/{nodesId}/deployments/{deploymentsId}/devices', http_method='POST', method_id='sasportal.nodes.deployments.devices.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha1/{+parent}/devices', request_field='sasPortalDevice', request_type_name='SasportalNodesDeploymentsDevicesCreateRequest', response_type_name='SasPortalDevice', supports_download=False)

    def CreateSigned(self, request, global_params=None):
        """Creates a signed device under a node or customer.

      Args:
        request: (SasportalNodesDeploymentsDevicesCreateSignedRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalDevice) The response message.
      """
        config = self.GetMethodConfig('CreateSigned')
        return self._RunMethod(config, request, global_params=global_params)
    CreateSigned.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/nodes/{nodesId}/deployments/{deploymentsId}/devices:createSigned', http_method='POST', method_id='sasportal.nodes.deployments.devices.createSigned', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha1/{+parent}/devices:createSigned', request_field='sasPortalCreateSignedDeviceRequest', request_type_name='SasportalNodesDeploymentsDevicesCreateSignedRequest', response_type_name='SasPortalDevice', supports_download=False)

    def List(self, request, global_params=None):
        """Lists devices under a node or customer.

      Args:
        request: (SasportalNodesDeploymentsDevicesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalListDevicesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/nodes/{nodesId}/deployments/{deploymentsId}/devices', http_method='GET', method_id='sasportal.nodes.deployments.devices.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/devices', request_field='', request_type_name='SasportalNodesDeploymentsDevicesListRequest', response_type_name='SasPortalListDevicesResponse', supports_download=False)
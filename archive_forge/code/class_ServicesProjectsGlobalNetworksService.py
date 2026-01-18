from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.servicenetworking.v1 import servicenetworking_v1_messages as messages
class ServicesProjectsGlobalNetworksService(base_api.BaseApiService):
    """Service class for the services_projects_global_networks resource."""
    _NAME = 'services_projects_global_networks'

    def __init__(self, client):
        super(ServicenetworkingV1.ServicesProjectsGlobalNetworksService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Service producers use this method to get the configuration of their connection including the import/export of custom routes and subnetwork routes with public IP.

      Args:
        request: (ServicenetworkingServicesProjectsGlobalNetworksGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ConsumerConfig) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/services/{servicesId}/projects/{projectsId}/global/networks/{networksId}', http_method='GET', method_id='servicenetworking.services.projects.global.networks.get', ordered_params=['name'], path_params=['name'], query_params=['includeUsedIpRanges'], relative_path='v1/{+name}', request_field='', request_type_name='ServicenetworkingServicesProjectsGlobalNetworksGetRequest', response_type_name='ConsumerConfig', supports_download=False)

    def GetVpcServiceControls(self, request, global_params=None):
        """Consumers use this method to find out the state of VPC Service Controls. The controls could be enabled or disabled for a connection.

      Args:
        request: (ServicenetworkingServicesProjectsGlobalNetworksGetVpcServiceControlsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (VpcServiceControls) The response message.
      """
        config = self.GetMethodConfig('GetVpcServiceControls')
        return self._RunMethod(config, request, global_params=global_params)
    GetVpcServiceControls.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/services/{servicesId}/projects/{projectsId}/global/networks/{networksId}/vpcServiceControls', http_method='GET', method_id='servicenetworking.services.projects.global.networks.getVpcServiceControls', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}/vpcServiceControls', request_field='', request_type_name='ServicenetworkingServicesProjectsGlobalNetworksGetVpcServiceControlsRequest', response_type_name='VpcServiceControls', supports_download=False)

    def UpdateConsumerConfig(self, request, global_params=None):
        """Service producers use this method to update the configuration of their connection including the import/export of custom routes and subnetwork routes with public IP.

      Args:
        request: (ServicenetworkingServicesProjectsGlobalNetworksUpdateConsumerConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('UpdateConsumerConfig')
        return self._RunMethod(config, request, global_params=global_params)
    UpdateConsumerConfig.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/services/{servicesId}/projects/{projectsId}/global/networks/{networksId}:updateConsumerConfig', http_method='PATCH', method_id='servicenetworking.services.projects.global.networks.updateConsumerConfig', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}:updateConsumerConfig', request_field='updateConsumerConfigRequest', request_type_name='ServicenetworkingServicesProjectsGlobalNetworksUpdateConsumerConfigRequest', response_type_name='Operation', supports_download=False)
from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.servicenetworking.v1 import servicenetworking_v1_messages as messages
class ServicesDnsZonesService(base_api.BaseApiService):
    """Service class for the services_dnsZones resource."""
    _NAME = 'services_dnsZones'

    def __init__(self, client):
        super(ServicenetworkingV1.ServicesDnsZonesService, self).__init__(client)
        self._upload_configs = {}

    def Add(self, request, global_params=None):
        """Service producers can use this method to add private DNS zones in the shared producer host project and matching peering zones in the consumer project.

      Args:
        request: (ServicenetworkingServicesDnsZonesAddRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Add')
        return self._RunMethod(config, request, global_params=global_params)
    Add.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/services/{servicesId}/dnsZones:add', http_method='POST', method_id='servicenetworking.services.dnsZones.add', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/dnsZones:add', request_field='addDnsZoneRequest', request_type_name='ServicenetworkingServicesDnsZonesAddRequest', response_type_name='Operation', supports_download=False)

    def Remove(self, request, global_params=None):
        """Service producers can use this method to remove private DNS zones in the shared producer host project and matching peering zones in the consumer project.

      Args:
        request: (ServicenetworkingServicesDnsZonesRemoveRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Remove')
        return self._RunMethod(config, request, global_params=global_params)
    Remove.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/services/{servicesId}/dnsZones:remove', http_method='POST', method_id='servicenetworking.services.dnsZones.remove', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/dnsZones:remove', request_field='removeDnsZoneRequest', request_type_name='ServicenetworkingServicesDnsZonesRemoveRequest', response_type_name='Operation', supports_download=False)
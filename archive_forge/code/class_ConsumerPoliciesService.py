from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.serviceusage.v2alpha import serviceusage_v2alpha_messages as messages
class ConsumerPoliciesService(base_api.BaseApiService):
    """Service class for the consumerPolicies resource."""
    _NAME = 'consumerPolicies'

    def __init__(self, client):
        super(ServiceusageV2alpha.ConsumerPoliciesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Get the consumer policy of a resource.

      Args:
        request: (ServiceusageConsumerPoliciesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleApiServiceusageV2alphaConsumerPolicy) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2alpha/{v2alphaId}/{v2alphaId1}/consumerPolicies/{consumerPoliciesId}', http_method='GET', method_id='serviceusage.consumerPolicies.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2alpha/{+name}', request_field='', request_type_name='ServiceusageConsumerPoliciesGetRequest', response_type_name='GoogleApiServiceusageV2alphaConsumerPolicy', supports_download=False)

    def Patch(self, request, global_params=None):
        """Update the consumer policy of a resource.

      Args:
        request: (ServiceusageConsumerPoliciesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2alpha/{v2alphaId}/{v2alphaId1}/consumerPolicies/{consumerPoliciesId}', http_method='PATCH', method_id='serviceusage.consumerPolicies.patch', ordered_params=['name'], path_params=['name'], query_params=['force', 'validateOnly'], relative_path='v2alpha/{+name}', request_field='googleApiServiceusageV2alphaConsumerPolicy', request_type_name='ServiceusageConsumerPoliciesPatchRequest', response_type_name='Operation', supports_download=False)
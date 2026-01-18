from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.serviceusage.v2alpha import serviceusage_v2alpha_messages as messages
class V2alphaService(base_api.BaseApiService):
    """Service class for the v2alpha resource."""
    _NAME = 'v2alpha'

    def __init__(self, client):
        super(ServiceusageV2alpha.V2alphaService, self).__init__(client)
        self._upload_configs = {}

    def GetEffectivePolicy(self, request, global_params=None):
        """Get effective consumer policy for a resource, which contains enable rule information of consumer policies from the resource hierarchy.

      Args:
        request: (ServiceusageGetEffectivePolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (EffectivePolicy) The response message.
      """
        config = self.GetMethodConfig('GetEffectivePolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetEffectivePolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2alpha/{v2alphaId}/{v2alphaId1}/effectivePolicy', http_method='GET', method_id='serviceusage.getEffectivePolicy', ordered_params=['name'], path_params=['name'], query_params=['view'], relative_path='v2alpha/{+name}', request_field='', request_type_name='ServiceusageGetEffectivePolicyRequest', response_type_name='EffectivePolicy', supports_download=False)

    def TestEnabled(self, request, global_params=None):
        """Tests a value against the result of merging consumer policies in the resource hierarchy. This operation is designed to be used for building policy-aware UIs and command-line tools, not for access checking.

      Args:
        request: (ServiceusageTestEnabledRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (State) The response message.
      """
        config = self.GetMethodConfig('TestEnabled')
        return self._RunMethod(config, request, global_params=global_params)
    TestEnabled.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2alpha/{v2alphaId}/{v2alphaId1}:testEnabled', http_method='POST', method_id='serviceusage.testEnabled', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2alpha/{+name}:testEnabled', request_field='testEnabledRequest', request_type_name='ServiceusageTestEnabledRequest', response_type_name='State', supports_download=False)
from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.binaryauthorization.v1alpha2 import binaryauthorization_v1alpha2_messages as messages
class SystempolicyService(base_api.BaseApiService):
    """Service class for the systempolicy resource."""
    _NAME = 'systempolicy'

    def __init__(self, client):
        super(BinaryauthorizationV1alpha2.SystempolicyService, self).__init__(client)
        self._upload_configs = {}

    def GetPolicy(self, request, global_params=None):
        """Gets the current system policy in the specified location.

      Args:
        request: (BinaryauthorizationSystempolicyGetPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/locations/{locationsId}/policy', http_method='GET', method_id='binaryauthorization.systempolicy.getPolicy', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='BinaryauthorizationSystempolicyGetPolicyRequest', response_type_name='Policy', supports_download=False)
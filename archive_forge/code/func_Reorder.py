from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.recaptchaenterprise.v1 import recaptchaenterprise_v1_messages as messages
def Reorder(self, request, global_params=None):
    """Reorders all firewall policies.

      Args:
        request: (RecaptchaenterpriseProjectsFirewallpoliciesReorderRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1ReorderFirewallPoliciesResponse) The response message.
      """
    config = self.GetMethodConfig('Reorder')
    return self._RunMethod(config, request, global_params=global_params)
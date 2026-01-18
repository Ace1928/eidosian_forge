from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmwareengine.v1 import vmwareengine_v1_messages as messages
def UpdateDnsForwarding(self, request, global_params=None):
    """Updates the parameters of the `DnsForwarding` config, like associated domains. Only fields specified in `update_mask` are applied.

      Args:
        request: (VmwareengineProjectsLocationsPrivateCloudsUpdateDnsForwardingRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('UpdateDnsForwarding')
    return self._RunMethod(config, request, global_params=global_params)
from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def GetDiagnostics(self, request, global_params=None):
    """Returns the interconnectDiagnostics for the specified Interconnect. In the event of a global outage, do not use this API to make decisions about where to redirect your network traffic. Unlike a VLAN attachment, which is regional, a Cloud Interconnect connection is a global resource. A global outage can prevent this API from functioning properly.

      Args:
        request: (ComputeInterconnectsGetDiagnosticsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InterconnectsGetDiagnosticsResponse) The response message.
      """
    config = self.GetMethodConfig('GetDiagnostics')
    return self._RunMethod(config, request, global_params=global_params)
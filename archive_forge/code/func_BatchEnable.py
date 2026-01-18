from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.serviceusage.v1 import serviceusage_v1_messages as messages
def BatchEnable(self, request, global_params=None):
    """Enable multiple services on a project. The operation is atomic: if enabling any service fails, then the entire batch fails, and no state changes occur. To enable a single service, use the `EnableService` method instead.

      Args:
        request: (ServiceusageServicesBatchEnableRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('BatchEnable')
    return self._RunMethod(config, request, global_params=global_params)
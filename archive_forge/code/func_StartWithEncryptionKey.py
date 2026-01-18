from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def StartWithEncryptionKey(self, request, global_params=None):
    """Starts an instance that was stopped using the instances().stop method. For more information, see Restart an instance.

      Args:
        request: (ComputeInstancesStartWithEncryptionKeyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('StartWithEncryptionKey')
    return self._RunMethod(config, request, global_params=global_params)
from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def StartAsyncReplication(self, request, global_params=None):
    """Starts asynchronous replication. Must be invoked on the primary disk.

      Args:
        request: (ComputeRegionDisksStartAsyncReplicationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('StartAsyncReplication')
    return self._RunMethod(config, request, global_params=global_params)
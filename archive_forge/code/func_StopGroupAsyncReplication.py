from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def StopGroupAsyncReplication(self, request, global_params=None):
    """Stops asynchronous replication for a consistency group of disks. Can be invoked either in the primary or secondary scope.

      Args:
        request: (ComputeRegionDisksStopGroupAsyncReplicationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('StopGroupAsyncReplication')
    return self._RunMethod(config, request, global_params=global_params)
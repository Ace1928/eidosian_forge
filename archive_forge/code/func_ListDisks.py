from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def ListDisks(self, request, global_params=None):
    """Lists the disks in a specified storage pool.

      Args:
        request: (ComputeStoragePoolsListDisksRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (StoragePoolListDisks) The response message.
      """
    config = self.GetMethodConfig('ListDisks')
    return self._RunMethod(config, request, global_params=global_params)
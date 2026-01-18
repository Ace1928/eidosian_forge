from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def BulkInsert(self, request, global_params=None):
    """Creates multiple instances in a given region. Count specifies the number of instances to create.

      Args:
        request: (ComputeRegionInstancesBulkInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('BulkInsert')
    return self._RunMethod(config, request, global_params=global_params)
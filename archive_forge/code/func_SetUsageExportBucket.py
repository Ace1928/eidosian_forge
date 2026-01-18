from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def SetUsageExportBucket(self, request, global_params=None):
    """Enables the usage export feature and sets the usage export bucket where reports are stored. If you provide an empty request body using this method, the usage export feature will be disabled.

      Args:
        request: (ComputeProjectsSetUsageExportBucketRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('SetUsageExportBucket')
    return self._RunMethod(config, request, global_params=global_params)
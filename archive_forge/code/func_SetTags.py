from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def SetTags(self, request, global_params=None):
    """Sets network tags for the specified instance to the data included in the request.

      Args:
        request: (ComputeInstancesSetTagsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('SetTags')
    return self._RunMethod(config, request, global_params=global_params)
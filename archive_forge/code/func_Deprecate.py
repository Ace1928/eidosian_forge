from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def Deprecate(self, request, global_params=None):
    """Sets the deprecation status of an image. If an empty request body is given, clears the deprecation status instead.

      Args:
        request: (ComputeImagesDeprecateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('Deprecate')
    return self._RunMethod(config, request, global_params=global_params)
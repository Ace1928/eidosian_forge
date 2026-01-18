from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def GetFromFamily(self, request, global_params=None):
    """Returns the latest image that is part of an image family and is not deprecated. For more information on image families, see Public image families documentation.

      Args:
        request: (ComputeImagesGetFromFamilyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Image) The response message.
      """
    config = self.GetMethodConfig('GetFromFamily')
    return self._RunMethod(config, request, global_params=global_params)
from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def GetPartnerMetadata(self, request, global_params=None):
    """Gets partner metadata of the specified instance and namespaces.

      Args:
        request: (ComputeInstancesGetPartnerMetadataRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PartnerMetadata) The response message.
      """
    config = self.GetMethodConfig('GetPartnerMetadata')
    return self._RunMethod(config, request, global_params=global_params)
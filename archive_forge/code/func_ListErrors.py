from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def ListErrors(self, request, global_params=None):
    """Lists all errors thrown by actions on instances for a given regional managed instance group. The filter and orderBy query parameters are not supported.

      Args:
        request: (ComputeRegionInstanceGroupManagersListErrorsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RegionInstanceGroupManagersListErrorsResponse) The response message.
      """
    config = self.GetMethodConfig('ListErrors')
    return self._RunMethod(config, request, global_params=global_params)
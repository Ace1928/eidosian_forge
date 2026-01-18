from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def SetMachineResources(self, request, global_params=None):
    """Changes the number and/or type of accelerator for a stopped instance to the values specified in the request.

      Args:
        request: (ComputeInstancesSetMachineResourcesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('SetMachineResources')
    return self._RunMethod(config, request, global_params=global_params)
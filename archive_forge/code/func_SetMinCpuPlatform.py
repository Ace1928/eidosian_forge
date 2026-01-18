from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def SetMinCpuPlatform(self, request, global_params=None):
    """Changes the minimum CPU platform that this instance should use. This method can only be called on a stopped instance. For more information, read Specifying a Minimum CPU Platform.

      Args:
        request: (ComputeInstancesSetMinCpuPlatformRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('SetMinCpuPlatform')
    return self._RunMethod(config, request, global_params=global_params)
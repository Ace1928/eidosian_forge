from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def ResizeAdvanced(self, request, global_params=None):
    """Resizes the regional managed instance group with advanced configuration options like disabling creation retries. This is an extended version of the resize method. If you increase the size, the group creates new instances using the current instance template. If you decrease the size, the group deletes one or more instances. The resize operation is marked DONE if the resize request is successful. The underlying actions take additional time. You must separately verify the status of the creating or deleting actions with the get or listmanagedinstances method. If the group is part of a backend service that has enabled connection draining, it can take up to 60 seconds after the connection draining duration has elapsed before the VM instance is removed or deleted.

      Args:
        request: (ComputeRegionInstanceGroupManagersResizeAdvancedRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('ResizeAdvanced')
    return self._RunMethod(config, request, global_params=global_params)
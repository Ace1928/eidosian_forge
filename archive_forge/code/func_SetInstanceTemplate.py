from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def SetInstanceTemplate(self, request, global_params=None):
    """Sets the instance template to use when creating new instances or recreating instances in this group. Existing instances are not affected.

      Args:
        request: (ComputeRegionInstanceGroupManagersSetInstanceTemplateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('SetInstanceTemplate')
    return self._RunMethod(config, request, global_params=global_params)
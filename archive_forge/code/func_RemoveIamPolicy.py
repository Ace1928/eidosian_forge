from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datafusion.v1beta1 import datafusion_v1beta1_messages as messages
def RemoveIamPolicy(self, request, global_params=None):
    """Remove IAM policy that is currently set on the given resource.

      Args:
        request: (DatafusionProjectsLocationsRemoveIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RemoveIamPolicyResponse) The response message.
      """
    config = self.GetMethodConfig('RemoveIamPolicy')
    return self._RunMethod(config, request, global_params=global_params)
from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iam.v3beta import iam_v3beta_messages as messages
def SearchTargetPolicyBindings(self, request, global_params=None):
    """Search policy bindings by target. Returns all policy binding objects bound directly to target.

      Args:
        request: (IamProjectsLocationsPolicyBindingsSearchTargetPolicyBindingsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV3betaSearchTargetPolicyBindingsResponse) The response message.
      """
    config = self.GetMethodConfig('SearchTargetPolicyBindings')
    return self._RunMethod(config, request, global_params=global_params)
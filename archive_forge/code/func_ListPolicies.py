from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.iam.v2 import iam_v2_messages as messages
def ListPolicies(self, request, global_params=None):
    """Retrieves the policies of the specified kind that are attached to a resource. The response lists only policy metadata. In particular, policy rules are omitted.

      Args:
        request: (IamPoliciesListPoliciesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleIamV2ListPoliciesResponse) The response message.
      """
    config = self.GetMethodConfig('ListPolicies')
    return self._RunMethod(config, request, global_params=global_params)
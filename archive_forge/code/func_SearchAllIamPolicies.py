from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudasset.v1 import cloudasset_v1_messages as messages
def SearchAllIamPolicies(self, request, global_params=None):
    """Searches all IAM policies within the specified scope, such as a project, folder, or organization. The caller must be granted the `cloudasset.assets.searchAllIamPolicies` permission on the desired scope, otherwise the request will be rejected.

      Args:
        request: (CloudassetSearchAllIamPoliciesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SearchAllIamPoliciesResponse) The response message.
      """
    config = self.GetMethodConfig('SearchAllIamPolicies')
    return self._RunMethod(config, request, global_params=global_params)
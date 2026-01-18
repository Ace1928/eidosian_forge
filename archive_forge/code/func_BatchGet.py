from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudasset.v1 import cloudasset_v1_messages as messages
def BatchGet(self, request, global_params=None):
    """Gets effective IAM policies for a batch of resources.

      Args:
        request: (CloudassetEffectiveIamPoliciesBatchGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BatchGetEffectiveIamPoliciesResponse) The response message.
      """
    config = self.GetMethodConfig('BatchGet')
    return self._RunMethod(config, request, global_params=global_params)
from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudasset.v1 import cloudasset_v1_messages as messages
def AnalyzeIamPolicy(self, request, global_params=None):
    """Analyzes IAM policies to answer which identities have what accesses on which resources.

      Args:
        request: (CloudassetAnalyzeIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AnalyzeIamPolicyResponse) The response message.
      """
    config = self.GetMethodConfig('AnalyzeIamPolicy')
    return self._RunMethod(config, request, global_params=global_params)
from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudasset.v1p4alpha1 import cloudasset_v1p4alpha1_messages as messages
Analyzes IAM policies based on the specified request. Returns.
a list of IamPolicyAnalysisResult matching the request.

      Args:
        request: (CloudassetAnalyzeIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AnalyzeIamPolicyResponse) The response message.
      
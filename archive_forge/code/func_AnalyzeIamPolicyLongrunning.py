from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudasset.v1 import cloudasset_v1_messages as messages
def AnalyzeIamPolicyLongrunning(self, request, global_params=None):
    """Analyzes IAM policies asynchronously to answer which identities have what accesses on which resources, and writes the analysis results to a Google Cloud Storage or a BigQuery destination. For Cloud Storage destination, the output format is the JSON format that represents a AnalyzeIamPolicyResponse. This method implements the google.longrunning.Operation, which allows you to track the operation status. We recommend intervals of at least 2 seconds with exponential backoff retry to poll the operation result. The metadata contains the metadata for the long-running operation.

      Args:
        request: (CloudassetAnalyzeIamPolicyLongrunningRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('AnalyzeIamPolicyLongrunning')
    return self._RunMethod(config, request, global_params=global_params)
from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudasset.v1 import cloudasset_v1_messages as messages
def AnalyzeMove(self, request, global_params=None):
    """Analyze moving a resource to a specified destination without kicking off the actual move. The analysis is best effort depending on the user's permissions of viewing different hierarchical policies and configurations. The policies and configuration are subject to change before the actual resource migration takes place.

      Args:
        request: (CloudassetAnalyzeMoveRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AnalyzeMoveResponse) The response message.
      """
    config = self.GetMethodConfig('AnalyzeMove')
    return self._RunMethod(config, request, global_params=global_params)
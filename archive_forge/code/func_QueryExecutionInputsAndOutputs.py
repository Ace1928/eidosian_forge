from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
def QueryExecutionInputsAndOutputs(self, request, global_params=None):
    """Obtains the set of input and output Artifacts for this Execution, in the form of LineageSubgraph that also contains the Execution and connecting Events.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresExecutionsQueryExecutionInputsAndOutputsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1LineageSubgraph) The response message.
      """
    config = self.GetMethodConfig('QueryExecutionInputsAndOutputs')
    return self._RunMethod(config, request, global_params=global_params)
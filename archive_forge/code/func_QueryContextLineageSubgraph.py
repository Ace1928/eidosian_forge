from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
def QueryContextLineageSubgraph(self, request, global_params=None):
    """Retrieves Artifacts and Executions within the specified Context, connected by Event edges and returned as a LineageSubgraph.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresContextsQueryContextLineageSubgraphRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1LineageSubgraph) The response message.
      """
    config = self.GetMethodConfig('QueryContextLineageSubgraph')
    return self._RunMethod(config, request, global_params=global_params)
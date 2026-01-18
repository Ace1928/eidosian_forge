from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
def Explain(self, request, global_params=None):
    """Perform an online explanation. If deployed_model_id is specified, the corresponding DeployModel must have explanation_spec populated. If deployed_model_id is not specified, all DeployedModels must have explanation_spec populated.

      Args:
        request: (AiplatformProjectsLocationsEndpointsExplainRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ExplainResponse) The response message.
      """
    config = self.GetMethodConfig('Explain')
    return self._RunMethod(config, request, global_params=global_params)
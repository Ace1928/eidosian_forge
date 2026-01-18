from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
def RawPredict(self, request, global_params=None):
    """Perform an online prediction with an arbitrary HTTP payload. The response includes the following HTTP headers: * `X-Vertex-AI-Endpoint-Id`: ID of the Endpoint that served this prediction. * `X-Vertex-AI-Deployed-Model-Id`: ID of the Endpoint's DeployedModel that served this prediction.

      Args:
        request: (AiplatformProjectsLocationsPublishersModelsRawPredictRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleApiHttpBody) The response message.
      """
    config = self.GetMethodConfig('RawPredict')
    return self._RunMethod(config, request, global_params=global_params)
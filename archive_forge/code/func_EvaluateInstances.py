from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1beta1 import aiplatform_v1beta1_messages as messages
def EvaluateInstances(self, request, global_params=None):
    """Evaluates instances based on a given metric.

      Args:
        request: (AiplatformProjectsLocationsEvaluateInstancesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1beta1EvaluateInstancesResponse) The response message.
      """
    config = self.GetMethodConfig('EvaluateInstances')
    return self._RunMethod(config, request, global_params=global_params)
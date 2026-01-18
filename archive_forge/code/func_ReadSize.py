from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
def ReadSize(self, request, global_params=None):
    """Returns the storage size for a given TensorBoard instance.

      Args:
        request: (AiplatformProjectsLocationsTensorboardsReadSizeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ReadTensorboardSizeResponse) The response message.
      """
    config = self.GetMethodConfig('ReadSize')
    return self._RunMethod(config, request, global_params=global_params)
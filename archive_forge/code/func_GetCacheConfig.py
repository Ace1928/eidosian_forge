from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1beta1 import aiplatform_v1beta1_messages as messages
def GetCacheConfig(self, request, global_params=None):
    """Gets a GenAI cache config.

      Args:
        request: (AiplatformProjectsGetCacheConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1beta1CacheConfig) The response message.
      """
    config = self.GetMethodConfig('GetCacheConfig')
    return self._RunMethod(config, request, global_params=global_params)
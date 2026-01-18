from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1beta1 import aiplatform_v1beta1_messages as messages
def UpdateCacheConfig(self, request, global_params=None):
    """Updates a cache config.

      Args:
        request: (GoogleCloudAiplatformV1beta1CacheConfig) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
    config = self.GetMethodConfig('UpdateCacheConfig')
    return self._RunMethod(config, request, global_params=global_params)
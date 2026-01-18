from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
def RemoveContextChildren(self, request, global_params=None):
    """Remove a set of children contexts from a parent Context. If any of the child Contexts were NOT added to the parent Context, they are simply skipped.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresContextsRemoveContextChildrenRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1RemoveContextChildrenResponse) The response message.
      """
    config = self.GetMethodConfig('RemoveContextChildren')
    return self._RunMethod(config, request, global_params=global_params)
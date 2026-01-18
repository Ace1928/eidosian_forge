from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
def UndeployIndex(self, request, global_params=None):
    """Undeploys an Index from an IndexEndpoint, removing a DeployedIndex from it, and freeing all resources it's using.

      Args:
        request: (AiplatformProjectsLocationsIndexEndpointsUndeployIndexRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
    config = self.GetMethodConfig('UndeployIndex')
    return self._RunMethod(config, request, global_params=global_params)
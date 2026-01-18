from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
def DeployIndex(self, request, global_params=None):
    """Deploys an Index into this IndexEndpoint, creating a DeployedIndex within it. Only non-empty Indexes can be deployed.

      Args:
        request: (AiplatformProjectsLocationsIndexEndpointsDeployIndexRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
    config = self.GetMethodConfig('DeployIndex')
    return self._RunMethod(config, request, global_params=global_params)
from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.deploymentmanager.alpha import deploymentmanager_alpha_messages as messages
def ListTypes(self, request, global_params=None):
    """Lists all the type info for a TypeProvider.

      Args:
        request: (DeploymentmanagerTypeProvidersListTypesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TypeProvidersListTypesResponse) The response message.
      """
    config = self.GetMethodConfig('ListTypes')
    return self._RunMethod(config, request, global_params=global_params)
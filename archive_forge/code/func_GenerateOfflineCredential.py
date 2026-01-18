from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.edgecontainer.v1beta import edgecontainer_v1beta_messages as messages
def GenerateOfflineCredential(self, request, global_params=None):
    """Generates an offline credential for a Cluster.

      Args:
        request: (EdgecontainerProjectsLocationsClustersGenerateOfflineCredentialRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GenerateOfflineCredentialResponse) The response message.
      """
    config = self.GetMethodConfig('GenerateOfflineCredential')
    return self._RunMethod(config, request, global_params=global_params)
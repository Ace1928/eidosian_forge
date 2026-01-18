from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkehub.v1beta import gkehub_v1beta_messages as messages
def GenerateConnectManifest(self, request, global_params=None):
    """Generates the manifest for deployment of the GKE connect agent. **This method is used internally by Google-provided libraries.** Most clients should not need to call this method directly.

      Args:
        request: (GkehubProjectsLocationsMembershipsGenerateConnectManifestRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GenerateConnectManifestResponse) The response message.
      """
    config = self.GetMethodConfig('GenerateConnectManifest')
    return self._RunMethod(config, request, global_params=global_params)
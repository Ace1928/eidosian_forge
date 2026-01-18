from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigeeregistry.v1 import apigeeregistry_v1_messages as messages
def ReplaceArtifact(self, request, global_params=None):
    """Used to replace a specified artifact.

      Args:
        request: (Artifact) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Artifact) The response message.
      """
    config = self.GetMethodConfig('ReplaceArtifact')
    return self._RunMethod(config, request, global_params=global_params)
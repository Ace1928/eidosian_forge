from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.run.v1 import run_v1_messages as messages
def ReplaceSecret(self, request, global_params=None):
    """Rpc to replace a secret. Only the spec, metadata labels, and annotations are modifiable. After the Update request, Cloud Run will work to make the 'status' match the requested 'spec'. May provide metadata.resourceVersion to enforce update from last read for optimistic concurrency control.

      Args:
        request: (RunProjectsLocationsSecretsReplaceSecretRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Secret) The response message.
      """
    config = self.GetMethodConfig('ReplaceSecret')
    return self._RunMethod(config, request, global_params=global_params)
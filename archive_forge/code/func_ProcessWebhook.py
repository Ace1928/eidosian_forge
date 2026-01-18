from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbuild.v2 import cloudbuild_v2_messages as messages
def ProcessWebhook(self, request, global_params=None):
    """ProcessWebhook is called by the external SCM for notifying of events.

      Args:
        request: (CloudbuildProjectsLocationsConnectionsProcessWebhookRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
    config = self.GetMethodConfig('ProcessWebhook')
    return self._RunMethod(config, request, global_params=global_params)
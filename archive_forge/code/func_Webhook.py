from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbuild.v2 import cloudbuild_v2_messages as messages
def Webhook(self, request, global_params=None):
    """Processes webhooks posted towards a WorkflowTrigger.

      Args:
        request: (CloudbuildProjectsLocationsWorkflowsWebhookRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ProcessWorkflowTriggerWebhookResponse) The response message.
      """
    config = self.GetMethodConfig('Webhook')
    return self._RunMethod(config, request, global_params=global_params)
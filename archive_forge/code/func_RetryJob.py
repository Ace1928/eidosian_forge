from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.clouddeploy.v1 import clouddeploy_v1_messages as messages
def RetryJob(self, request, global_params=None):
    """Retries the specified Job in a Rollout.

      Args:
        request: (ClouddeployProjectsLocationsDeliveryPipelinesReleasesRolloutsRetryJobRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RetryJobResponse) The response message.
      """
    config = self.GetMethodConfig('RetryJob')
    return self._RunMethod(config, request, global_params=global_params)
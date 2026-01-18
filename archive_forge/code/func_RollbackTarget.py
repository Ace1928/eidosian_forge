from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.clouddeploy.v1 import clouddeploy_v1_messages as messages
def RollbackTarget(self, request, global_params=None):
    """Creates a `Rollout` to roll back the specified target.

      Args:
        request: (ClouddeployProjectsLocationsDeliveryPipelinesRollbackTargetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RollbackTargetResponse) The response message.
      """
    config = self.GetMethodConfig('RollbackTarget')
    return self._RunMethod(config, request, global_params=global_params)
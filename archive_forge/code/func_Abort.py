from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.configdelivery.v1alpha import configdelivery_v1alpha_messages as messages
def Abort(self, request, global_params=None):
    """Abort a Rollout.

      Args:
        request: (ConfigdeliveryProjectsLocationsFleetPackagesRolloutsAbortRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('Abort')
    return self._RunMethod(config, request, global_params=global_params)
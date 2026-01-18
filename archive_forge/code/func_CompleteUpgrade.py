from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.container.v1 import container_v1_messages as messages
def CompleteUpgrade(self, request, global_params=None):
    """CompleteNodePoolUpgrade will signal an on-going node pool upgrade to complete.

      Args:
        request: (ContainerProjectsLocationsClustersNodePoolsCompleteUpgradeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
    config = self.GetMethodConfig('CompleteUpgrade')
    return self._RunMethod(config, request, global_params=global_params)
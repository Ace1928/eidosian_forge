from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.file.v1beta1 import file_v1beta1_messages as messages
def PromoteReplica(self, request, global_params=None):
    """Promote an standby instance (replica).

      Args:
        request: (FileProjectsLocationsInstancesPromoteReplicaRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('PromoteReplica')
    return self._RunMethod(config, request, global_params=global_params)
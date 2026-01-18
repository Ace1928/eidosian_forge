from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sqladmin.v1beta4 import sqladmin_v1beta4_messages as messages
def ResetReplicaSize(self, request, global_params=None):
    """Reset Replica Size to primary instance disk size.

      Args:
        request: (SqlProjectsInstancesResetReplicaSizeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('ResetReplicaSize')
    return self._RunMethod(config, request, global_params=global_params)
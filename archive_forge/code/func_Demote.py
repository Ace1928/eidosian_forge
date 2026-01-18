from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sqladmin.v1beta4 import sqladmin_v1beta4_messages as messages
def Demote(self, request, global_params=None):
    """Demotes an existing standalone instance to be a Cloud SQL read replica for an external database server.

      Args:
        request: (SqlInstancesDemoteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('Demote')
    return self._RunMethod(config, request, global_params=global_params)
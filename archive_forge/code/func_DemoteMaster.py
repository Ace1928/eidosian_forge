from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sqladmin.v1beta4 import sqladmin_v1beta4_messages as messages
def DemoteMaster(self, request, global_params=None):
    """Demotes the stand-alone instance to be a Cloud SQL read replica for an external database server.

      Args:
        request: (SqlInstancesDemoteMasterRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('DemoteMaster')
    return self._RunMethod(config, request, global_params=global_params)
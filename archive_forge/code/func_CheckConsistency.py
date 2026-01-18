from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.bigtableadmin.v2 import bigtableadmin_v2_messages as messages
def CheckConsistency(self, request, global_params=None):
    """Checks replication consistency based on a consistency token, that is, if replication has caught up based on the conditions specified in the token and the check request.

      Args:
        request: (BigtableadminProjectsInstancesTablesCheckConsistencyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CheckConsistencyResponse) The response message.
      """
    config = self.GetMethodConfig('CheckConsistency')
    return self._RunMethod(config, request, global_params=global_params)
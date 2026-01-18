from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.bigtableadmin.v2 import bigtableadmin_v2_messages as messages
def GenerateConsistencyToken(self, request, global_params=None):
    """Generates a consistency token for a Table, which can be used in CheckConsistency to check whether mutations to the table that finished before this call started have been replicated. The tokens will be available for 90 days.

      Args:
        request: (BigtableadminProjectsInstancesTablesGenerateConsistencyTokenRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GenerateConsistencyTokenResponse) The response message.
      """
    config = self.GetMethodConfig('GenerateConsistencyToken')
    return self._RunMethod(config, request, global_params=global_params)
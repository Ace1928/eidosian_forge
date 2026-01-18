from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmmigration.v1 import vmmigration_v1_messages as messages
def FinalizeMigration(self, request, global_params=None):
    """Marks a migration as completed, deleting migration resources that are no longer being used. Only applicable after cutover is done.

      Args:
        request: (VmmigrationProjectsLocationsSourcesMigratingVmsFinalizeMigrationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('FinalizeMigration')
    return self._RunMethod(config, request, global_params=global_params)
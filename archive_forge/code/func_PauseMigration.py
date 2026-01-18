from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmmigration.v1 import vmmigration_v1_messages as messages
def PauseMigration(self, request, global_params=None):
    """Pauses a migration for a VM. If cycle tasks are running they will be cancelled, preserving source task data. Further replication cycles will not be triggered while the VM is paused.

      Args:
        request: (VmmigrationProjectsLocationsSourcesMigratingVmsPauseMigrationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('PauseMigration')
    return self._RunMethod(config, request, global_params=global_params)
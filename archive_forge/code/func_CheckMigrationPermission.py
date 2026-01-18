from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.managedidentities.v1 import managedidentities_v1_messages as messages
def CheckMigrationPermission(self, request, global_params=None):
    """CheckMigrationPermission API gets the current state of DomainMigration.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalDomainsCheckMigrationPermissionRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CheckMigrationPermissionResponse) The response message.
      """
    config = self.GetMethodConfig('CheckMigrationPermission')
    return self._RunMethod(config, request, global_params=global_params)
from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.managedidentities.v1 import managedidentities_v1_messages as messages
def EnableMigration(self, request, global_params=None):
    """Enable Domain Migration.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalDomainsEnableMigrationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('EnableMigration')
    return self._RunMethod(config, request, global_params=global_params)
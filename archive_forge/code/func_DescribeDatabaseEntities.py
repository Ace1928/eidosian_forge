from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datamigration.v1 import datamigration_v1_messages as messages
def DescribeDatabaseEntities(self, request, global_params=None):
    """Describes the database entities tree for a specific conversion workspace and a specific tree type. Database entities are not resources like conversion workspaces or mapping rules, and they can't be created, updated or deleted. Instead, they are simple data objects describing the structure of the client database.

      Args:
        request: (DatamigrationProjectsLocationsConversionWorkspacesDescribeDatabaseEntitiesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DescribeDatabaseEntitiesResponse) The response message.
      """
    config = self.GetMethodConfig('DescribeDatabaseEntities')
    return self._RunMethod(config, request, global_params=global_params)
from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datastore.v1 import datastore_v1_messages as messages
def AllocateIds(self, request, global_params=None):
    """Allocates IDs for the given keys, which is useful for referencing an entity before it is inserted.

      Args:
        request: (DatastoreProjectsAllocateIdsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AllocateIdsResponse) The response message.
      """
    config = self.GetMethodConfig('AllocateIds')
    return self._RunMethod(config, request, global_params=global_params)
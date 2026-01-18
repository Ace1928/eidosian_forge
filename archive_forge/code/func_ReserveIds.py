from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datastore.v1 import datastore_v1_messages as messages
def ReserveIds(self, request, global_params=None):
    """Prevents the supplied keys' IDs from being auto-allocated by Cloud Datastore.

      Args:
        request: (DatastoreProjectsReserveIdsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ReserveIdsResponse) The response message.
      """
    config = self.GetMethodConfig('ReserveIds')
    return self._RunMethod(config, request, global_params=global_params)
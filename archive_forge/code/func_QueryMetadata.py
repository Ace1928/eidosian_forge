from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.metastore.v1beta import metastore_v1beta_messages as messages
def QueryMetadata(self, request, global_params=None):
    """Query Dataproc Metastore metadata.

      Args:
        request: (MetastoreProjectsLocationsServicesQueryMetadataRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('QueryMetadata')
    return self._RunMethod(config, request, global_params=global_params)
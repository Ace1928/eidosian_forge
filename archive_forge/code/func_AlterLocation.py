from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.metastore.v1beta import metastore_v1beta_messages as messages
def AlterLocation(self, request, global_params=None):
    """Alter metadata resource location. The metadata resource can be a database, table, or partition. This functionality only updates the parent directory for the respective metadata resource and does not transfer any existing data to the new location.

      Args:
        request: (MetastoreProjectsLocationsServicesAlterLocationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('AlterLocation')
    return self._RunMethod(config, request, global_params=global_params)
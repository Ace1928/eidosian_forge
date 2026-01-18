from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.bigquerydatatransfer.v1 import bigquerydatatransfer_v1_messages as messages
def UnenrollDataSources(self, request, global_params=None):
    """Unenroll data sources in a user project. This allows users to remove transfer configurations for these data sources. They will no longer appear in the ListDataSources RPC and will also no longer appear in the [BigQuery UI](https://console.cloud.google.com/bigquery). Data transfers configurations of unenrolled data sources will not be scheduled.

      Args:
        request: (BigquerydatatransferProjectsLocationsUnenrollDataSourcesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
    config = self.GetMethodConfig('UnenrollDataSources')
    return self._RunMethod(config, request, global_params=global_params)
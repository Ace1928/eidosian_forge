from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.storageinsights.v1 import storageinsights_v1_messages as messages
def UnlinkDataset(self, request, global_params=None):
    """UnlinkDataset method for the projects_locations_datasetConfigs service.

      Args:
        request: (StorageinsightsProjectsLocationsDatasetConfigsUnlinkDatasetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('UnlinkDataset')
    return self._RunMethod(config, request, global_params=global_params)
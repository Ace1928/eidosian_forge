from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.parallelstore.v1beta import parallelstore_v1beta_messages as messages
def ImportData(self, request, global_params=None):
    """ImportData copies data from Cloud Storage to Parallelstore.

      Args:
        request: (ParallelstoreProjectsLocationsInstancesImportDataRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('ImportData')
    return self._RunMethod(config, request, global_params=global_params)
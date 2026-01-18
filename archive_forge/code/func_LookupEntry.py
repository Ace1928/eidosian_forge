from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataplex.v1 import dataplex_v1_messages as messages
def LookupEntry(self, request, global_params=None):
    """Looks up a single entry.

      Args:
        request: (DataplexProjectsLocationsLookupEntryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDataplexV1Entry) The response message.
      """
    config = self.GetMethodConfig('LookupEntry')
    return self._RunMethod(config, request, global_params=global_params)
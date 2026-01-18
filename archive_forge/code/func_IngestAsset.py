from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudasset.v1 import cloudasset_v1_messages as messages
def IngestAsset(self, request, global_params=None):
    """Ingests a 3rd party asset into CAIS.

      Args:
        request: (CloudassetIngestAssetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (IngestAssetResponse) The response message.
      """
    config = self.GetMethodConfig('IngestAsset')
    return self._RunMethod(config, request, global_params=global_params)
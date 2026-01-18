from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.translate.v3 import translate_v3_messages as messages
def ImportAdaptiveMtFile(self, request, global_params=None):
    """Imports an AdaptiveMtFile and adds all of its sentences into the AdaptiveMtDataset.

      Args:
        request: (TranslateProjectsLocationsAdaptiveMtDatasetsImportAdaptiveMtFileRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ImportAdaptiveMtFileResponse) The response message.
      """
    config = self.GetMethodConfig('ImportAdaptiveMtFile')
    return self._RunMethod(config, request, global_params=global_params)
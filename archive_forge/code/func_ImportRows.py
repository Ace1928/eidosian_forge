from __future__ import absolute_import
from apitools.base.py import base_api
from samples.fusiontables_sample.fusiontables_v1 import fusiontables_v1_messages as messages
def ImportRows(self, request, global_params=None, upload=None):
    """Import more rows into a table.

      Args:
        request: (FusiontablesTableImportRowsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
        upload: (Upload, default: None) If present, upload
            this stream with the request.
      Returns:
        (Import) The response message.
      """
    config = self.GetMethodConfig('ImportRows')
    upload_config = self.GetUploadConfig('ImportRows')
    return self._RunMethod(config, request, global_params=global_params, upload=upload, upload_config=upload_config)
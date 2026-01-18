from __future__ import absolute_import
from apitools.base.py import base_api
from samples.bigquery_sample.bigquery_v2 import bigquery_v2_messages as messages
def InsertAll(self, request, global_params=None):
    """Streams data into BigQuery one record at a time without needing to run a load job. Requires the WRITER dataset role.

      Args:
        request: (BigqueryTabledataInsertAllRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TableDataInsertAllResponse) The response message.
      """
    config = self.GetMethodConfig('InsertAll')
    return self._RunMethod(config, request, global_params=global_params)
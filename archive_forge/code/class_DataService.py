from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.logging.v2 import logging_v2_messages as messages
class DataService(base_api.BaseApiService):
    """Service class for the data resource."""
    _NAME = 'data'

    def __init__(self, client):
        super(LoggingV2.DataService, self).__init__(client)
        self._upload_configs = {}

    def QueryLocal(self, request, global_params=None):
        """Runs a (possibly multi-step) SQL query asynchronously in the customer project and returns handles that can be used to fetch the results of each step. View references are translated to linked dataset tables, and references to other raw BigQuery tables are permitted.

      Args:
        request: (QueryDataLocalRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (QueryDataResponse) The response message.
      """
        config = self.GetMethodConfig('QueryLocal')
        return self._RunMethod(config, request, global_params=global_params)
    QueryLocal.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='logging.data.queryLocal', ordered_params=[], path_params=[], query_params=[], relative_path='v2/data:queryLocal', request_field='<request>', request_type_name='QueryDataLocalRequest', response_type_name='QueryDataResponse', supports_download=False)
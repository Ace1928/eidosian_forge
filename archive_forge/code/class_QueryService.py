from __future__ import absolute_import
from apitools.base.py import base_api
from samples.fusiontables_sample.fusiontables_v1 import fusiontables_v1_messages as messages
class QueryService(base_api.BaseApiService):
    """Service class for the query resource."""
    _NAME = u'query'

    def __init__(self, client):
        super(FusiontablesV1.QueryService, self).__init__(client)
        self._upload_configs = {}

    def Sql(self, request, global_params=None, download=None):
        """Executes an SQL SELECT/INSERT/UPDATE/DELETE/SHOW/DESCRIBE/CREATE statement.

      Args:
        request: (FusiontablesQuerySqlRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
        download: (Download, default: None) If present, download
            data from the request via this stream.
      Returns:
        (Sqlresponse) The response message.
      """
        config = self.GetMethodConfig('Sql')
        return self._RunMethod(config, request, global_params=global_params, download=download)
    Sql.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'fusiontables.query.sql', ordered_params=[u'sql'], path_params=[], query_params=[u'hdrs', u'sql', u'typed'], relative_path=u'query', request_field='', request_type_name=u'FusiontablesQuerySqlRequest', response_type_name=u'Sqlresponse', supports_download=True)

    def SqlGet(self, request, global_params=None, download=None):
        """Executes an SQL SELECT/SHOW/DESCRIBE statement.

      Args:
        request: (FusiontablesQuerySqlGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
        download: (Download, default: None) If present, download
            data from the request via this stream.
      Returns:
        (Sqlresponse) The response message.
      """
        config = self.GetMethodConfig('SqlGet')
        return self._RunMethod(config, request, global_params=global_params, download=download)
    SqlGet.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'fusiontables.query.sqlGet', ordered_params=[u'sql'], path_params=[], query_params=[u'hdrs', u'sql', u'typed'], relative_path=u'query', request_field='', request_type_name=u'FusiontablesQuerySqlGetRequest', response_type_name=u'Sqlresponse', supports_download=True)
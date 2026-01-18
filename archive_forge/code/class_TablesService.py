from __future__ import absolute_import
from apitools.base.py import base_api
from samples.bigquery_sample.bigquery_v2 import bigquery_v2_messages as messages
class TablesService(base_api.BaseApiService):
    """Service class for the tables resource."""
    _NAME = u'tables'

    def __init__(self, client):
        super(BigqueryV2.TablesService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes the table specified by tableId from the dataset. If the table contains data, all the data will be deleted.

      Args:
        request: (BigqueryTablesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BigqueryTablesDeleteResponse) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method=u'DELETE', method_id=u'bigquery.tables.delete', ordered_params=[u'projectId', u'datasetId', u'tableId'], path_params=[u'datasetId', u'projectId', u'tableId'], query_params=[], relative_path=u'projects/{projectId}/datasets/{datasetId}/tables/{tableId}', request_field='', request_type_name=u'BigqueryTablesDeleteRequest', response_type_name=u'BigqueryTablesDeleteResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the specified table resource by table ID. This method does not return the data in the table, it only returns the table resource, which describes the structure of this table.

      Args:
        request: (BigqueryTablesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Table) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'bigquery.tables.get', ordered_params=[u'projectId', u'datasetId', u'tableId'], path_params=[u'datasetId', u'projectId', u'tableId'], query_params=[], relative_path=u'projects/{projectId}/datasets/{datasetId}/tables/{tableId}', request_field='', request_type_name=u'BigqueryTablesGetRequest', response_type_name=u'Table', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a new, empty table in the dataset.

      Args:
        request: (BigqueryTablesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Table) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'bigquery.tables.insert', ordered_params=[u'projectId', u'datasetId'], path_params=[u'datasetId', u'projectId'], query_params=[], relative_path=u'projects/{projectId}/datasets/{datasetId}/tables', request_field=u'table', request_type_name=u'BigqueryTablesInsertRequest', response_type_name=u'Table', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all tables in the specified dataset. Requires the READER dataset role.

      Args:
        request: (BigqueryTablesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TableList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'bigquery.tables.list', ordered_params=[u'projectId', u'datasetId'], path_params=[u'datasetId', u'projectId'], query_params=[u'maxResults', u'pageToken'], relative_path=u'projects/{projectId}/datasets/{datasetId}/tables', request_field='', request_type_name=u'BigqueryTablesListRequest', response_type_name=u'TableList', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates information in an existing table. The update method replaces the entire table resource, whereas the patch method only replaces fields that are provided in the submitted table resource. This method supports patch semantics.

      Args:
        request: (BigqueryTablesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Table) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PATCH', method_id=u'bigquery.tables.patch', ordered_params=[u'projectId', u'datasetId', u'tableId'], path_params=[u'datasetId', u'projectId', u'tableId'], query_params=[], relative_path=u'projects/{projectId}/datasets/{datasetId}/tables/{tableId}', request_field=u'table', request_type_name=u'BigqueryTablesPatchRequest', response_type_name=u'Table', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates information in an existing table. The update method replaces the entire table resource, whereas the patch method only replaces fields that are provided in the submitted table resource.

      Args:
        request: (BigqueryTablesUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Table) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PUT', method_id=u'bigquery.tables.update', ordered_params=[u'projectId', u'datasetId', u'tableId'], path_params=[u'datasetId', u'projectId', u'tableId'], query_params=[], relative_path=u'projects/{projectId}/datasets/{datasetId}/tables/{tableId}', request_field=u'table', request_type_name=u'BigqueryTablesUpdateRequest', response_type_name=u'Table', supports_download=False)
from __future__ import absolute_import
from apitools.base.py import base_api
from samples.fusiontables_sample.fusiontables_v1 import fusiontables_v1_messages as messages
class TableService(base_api.BaseApiService):
    """Service class for the table resource."""
    _NAME = u'table'

    def __init__(self, client):
        super(FusiontablesV1.TableService, self).__init__(client)
        self._upload_configs = {'ImportRows': base_api.ApiUploadInfo(accept=['application/octet-stream'], max_size=262144000, resumable_multipart=True, resumable_path=u'/resumable/upload/fusiontables/v1/tables/{tableId}/import', simple_multipart=True, simple_path=u'/upload/fusiontables/v1/tables/{tableId}/import'), 'ImportTable': base_api.ApiUploadInfo(accept=['application/octet-stream'], max_size=262144000, resumable_multipart=True, resumable_path=u'/resumable/upload/fusiontables/v1/tables/import', simple_multipart=True, simple_path=u'/upload/fusiontables/v1/tables/import')}

    def Copy(self, request, global_params=None):
        """Copies a table.

      Args:
        request: (FusiontablesTableCopyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Table) The response message.
      """
        config = self.GetMethodConfig('Copy')
        return self._RunMethod(config, request, global_params=global_params)
    Copy.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'fusiontables.table.copy', ordered_params=[u'tableId'], path_params=[u'tableId'], query_params=[u'copyPresentation'], relative_path=u'tables/{tableId}/copy', request_field='', request_type_name=u'FusiontablesTableCopyRequest', response_type_name=u'Table', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a table.

      Args:
        request: (FusiontablesTableDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FusiontablesTableDeleteResponse) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method=u'DELETE', method_id=u'fusiontables.table.delete', ordered_params=[u'tableId'], path_params=[u'tableId'], query_params=[], relative_path=u'tables/{tableId}', request_field='', request_type_name=u'FusiontablesTableDeleteRequest', response_type_name=u'FusiontablesTableDeleteResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves a specific table by its id.

      Args:
        request: (FusiontablesTableGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Table) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'fusiontables.table.get', ordered_params=[u'tableId'], path_params=[u'tableId'], query_params=[], relative_path=u'tables/{tableId}', request_field='', request_type_name=u'FusiontablesTableGetRequest', response_type_name=u'Table', supports_download=False)

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
    ImportRows.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'fusiontables.table.importRows', ordered_params=[u'tableId'], path_params=[u'tableId'], query_params=[u'delimiter', u'encoding', u'endLine', u'isStrict', u'startLine'], relative_path=u'tables/{tableId}/import', request_field='', request_type_name=u'FusiontablesTableImportRowsRequest', response_type_name=u'Import', supports_download=False)

    def ImportTable(self, request, global_params=None, upload=None):
        """Import a new table.

      Args:
        request: (FusiontablesTableImportTableRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
        upload: (Upload, default: None) If present, upload
            this stream with the request.
      Returns:
        (Table) The response message.
      """
        config = self.GetMethodConfig('ImportTable')
        upload_config = self.GetUploadConfig('ImportTable')
        return self._RunMethod(config, request, global_params=global_params, upload=upload, upload_config=upload_config)
    ImportTable.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'fusiontables.table.importTable', ordered_params=[u'name'], path_params=[], query_params=[u'delimiter', u'encoding', u'name'], relative_path=u'tables/import', request_field='', request_type_name=u'FusiontablesTableImportTableRequest', response_type_name=u'Table', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a new table.

      Args:
        request: (Table) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Table) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'fusiontables.table.insert', ordered_params=[], path_params=[], query_params=[], relative_path=u'tables', request_field='<request>', request_type_name=u'Table', response_type_name=u'Table', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of tables a user owns.

      Args:
        request: (FusiontablesTableListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TableList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'fusiontables.table.list', ordered_params=[], path_params=[], query_params=[u'maxResults', u'pageToken'], relative_path=u'tables', request_field='', request_type_name=u'FusiontablesTableListRequest', response_type_name=u'TableList', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an existing table. Unless explicitly requested, only the name, description, and attribution will be updated. This method supports patch semantics.

      Args:
        request: (FusiontablesTablePatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Table) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PATCH', method_id=u'fusiontables.table.patch', ordered_params=[u'tableId'], path_params=[u'tableId'], query_params=[u'replaceViewDefinition'], relative_path=u'tables/{tableId}', request_field=u'table', request_type_name=u'FusiontablesTablePatchRequest', response_type_name=u'Table', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates an existing table. Unless explicitly requested, only the name, description, and attribution will be updated.

      Args:
        request: (FusiontablesTableUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Table) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PUT', method_id=u'fusiontables.table.update', ordered_params=[u'tableId'], path_params=[u'tableId'], query_params=[u'replaceViewDefinition'], relative_path=u'tables/{tableId}', request_field=u'table', request_type_name=u'FusiontablesTableUpdateRequest', response_type_name=u'Table', supports_download=False)
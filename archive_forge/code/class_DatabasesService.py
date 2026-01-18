from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sqladmin.v1beta4 import sqladmin_v1beta4_messages as messages
class DatabasesService(base_api.BaseApiService):
    """Service class for the databases resource."""
    _NAME = 'databases'

    def __init__(self, client):
        super(SqladminV1beta4.DatabasesService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes a database from a Cloud SQL instance.

      Args:
        request: (SqlDatabasesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='sql.databases.delete', ordered_params=['project', 'instance', 'database'], path_params=['database', 'instance', 'project'], query_params=[], relative_path='sql/v1beta4/projects/{project}/instances/{instance}/databases/{database}', request_field='', request_type_name='SqlDatabasesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves a resource containing information about a database inside a Cloud SQL instance.

      Args:
        request: (SqlDatabasesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Database) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='sql.databases.get', ordered_params=['project', 'instance', 'database'], path_params=['database', 'instance', 'project'], query_params=[], relative_path='sql/v1beta4/projects/{project}/instances/{instance}/databases/{database}', request_field='', request_type_name='SqlDatabasesGetRequest', response_type_name='Database', supports_download=False)

    def Insert(self, request, global_params=None):
        """Inserts a resource containing information about a database inside a Cloud SQL instance. **Note:** You can't modify the default character set and collation.

      Args:
        request: (Database) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='sql.databases.insert', ordered_params=['project', 'instance'], path_params=['instance', 'project'], query_params=[], relative_path='sql/v1beta4/projects/{project}/instances/{instance}/databases', request_field='<request>', request_type_name='Database', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists databases in the specified Cloud SQL instance.

      Args:
        request: (SqlDatabasesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DatabasesListResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='sql.databases.list', ordered_params=['project', 'instance'], path_params=['instance', 'project'], query_params=[], relative_path='sql/v1beta4/projects/{project}/instances/{instance}/databases', request_field='', request_type_name='SqlDatabasesListRequest', response_type_name='DatabasesListResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Partially updates a resource containing information about a database inside a Cloud SQL instance. This method supports patch semantics.

      Args:
        request: (SqlDatabasesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='sql.databases.patch', ordered_params=['project', 'instance', 'database'], path_params=['database', 'instance', 'project'], query_params=[], relative_path='sql/v1beta4/projects/{project}/instances/{instance}/databases/{database}', request_field='databaseResource', request_type_name='SqlDatabasesPatchRequest', response_type_name='Operation', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates a resource containing information about a database inside a Cloud SQL instance.

      Args:
        request: (SqlDatabasesUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method='PUT', method_id='sql.databases.update', ordered_params=['project', 'instance', 'database'], path_params=['database', 'instance', 'project'], query_params=[], relative_path='sql/v1beta4/projects/{project}/instances/{instance}/databases/{database}', request_field='databaseResource', request_type_name='SqlDatabasesUpdateRequest', response_type_name='Operation', supports_download=False)
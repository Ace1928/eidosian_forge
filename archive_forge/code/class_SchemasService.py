from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.admin.v1 import admin_v1_messages as messages
class SchemasService(base_api.BaseApiService):
    """Service class for the schemas resource."""
    _NAME = u'schemas'

    def __init__(self, client):
        super(AdminDirectoryV1.SchemasService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Delete schema.

      Args:
        request: (DirectorySchemasDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (DirectorySchemasDeleteResponse) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method=u'DELETE', method_id=u'directory.schemas.delete', ordered_params=[u'customerId', u'schemaKey'], path_params=[u'customerId', u'schemaKey'], query_params=[], relative_path=u'customer/{customerId}/schemas/{schemaKey}', request_field='', request_type_name=u'DirectorySchemasDeleteRequest', response_type_name=u'DirectorySchemasDeleteResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieve schema.

      Args:
        request: (DirectorySchemasGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (Schema) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'directory.schemas.get', ordered_params=[u'customerId', u'schemaKey'], path_params=[u'customerId', u'schemaKey'], query_params=[], relative_path=u'customer/{customerId}/schemas/{schemaKey}', request_field='', request_type_name=u'DirectorySchemasGetRequest', response_type_name=u'Schema', supports_download=False)

    def Insert(self, request, global_params=None):
        """Create schema.

      Args:
        request: (DirectorySchemasInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (Schema) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'directory.schemas.insert', ordered_params=[u'customerId'], path_params=[u'customerId'], query_params=[], relative_path=u'customer/{customerId}/schemas', request_field=u'schema', request_type_name=u'DirectorySchemasInsertRequest', response_type_name=u'Schema', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieve all schemas for a customer.

      Args:
        request: (DirectorySchemasListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (Schemas) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'directory.schemas.list', ordered_params=[u'customerId'], path_params=[u'customerId'], query_params=[], relative_path=u'customer/{customerId}/schemas', request_field='', request_type_name=u'DirectorySchemasListRequest', response_type_name=u'Schemas', supports_download=False)

    def Patch(self, request, global_params=None):
        """Update schema.

      This method supports patch semantics.

      Args:
        request: (DirectorySchemasPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (Schema) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PATCH', method_id=u'directory.schemas.patch', ordered_params=[u'customerId', u'schemaKey'], path_params=[u'customerId', u'schemaKey'], query_params=[], relative_path=u'customer/{customerId}/schemas/{schemaKey}', request_field=u'schema', request_type_name=u'DirectorySchemasPatchRequest', response_type_name=u'Schema', supports_download=False)

    def Update(self, request, global_params=None):
        """Update schema.

      Args:
        request: (DirectorySchemasUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (Schema) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PUT', method_id=u'directory.schemas.update', ordered_params=[u'customerId', u'schemaKey'], path_params=[u'customerId', u'schemaKey'], query_params=[], relative_path=u'customer/{customerId}/schemas/{schemaKey}', request_field=u'schema', request_type_name=u'DirectorySchemasUpdateRequest', response_type_name=u'Schema', supports_download=False)
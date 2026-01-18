import platform
import sys
from apitools.base.py import base_api
import gslib
from gslib.metrics import MetricsCollector
from gslib.third_party.storage_apitools import storage_v1_messages as messages
class ObjectAccessControlsService(base_api.BaseApiService):
    """Service class for the objectAccessControls resource."""
    _NAME = u'objectAccessControls'

    def __init__(self, client):
        super(StorageV1.ObjectAccessControlsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Permanently deletes the ACL entry for the specified entity on the specified object.

      Args:
        request: (StorageObjectAccessControlsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (StorageObjectAccessControlsDeleteResponse) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method=u'DELETE', method_id=u'storage.objectAccessControls.delete', ordered_params=[u'bucket', u'object', u'entity'], path_params=[u'bucket', u'entity', u'object'], query_params=[u'generation', u'userProject'], relative_path=u'b/{bucket}/o/{object}/acl/{entity}', request_field='', request_type_name=u'StorageObjectAccessControlsDeleteRequest', response_type_name=u'StorageObjectAccessControlsDeleteResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the ACL entry for the specified entity on the specified object.

      Args:
        request: (StorageObjectAccessControlsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ObjectAccessControl) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'storage.objectAccessControls.get', ordered_params=[u'bucket', u'object', u'entity'], path_params=[u'bucket', u'entity', u'object'], query_params=[u'generation', u'userProject'], relative_path=u'b/{bucket}/o/{object}/acl/{entity}', request_field='', request_type_name=u'StorageObjectAccessControlsGetRequest', response_type_name=u'ObjectAccessControl', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a new ACL entry on the specified object.

      Args:
        request: (StorageObjectAccessControlsInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ObjectAccessControl) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'storage.objectAccessControls.insert', ordered_params=[u'bucket', u'object'], path_params=[u'bucket', u'object'], query_params=[u'generation', u'userProject'], relative_path=u'b/{bucket}/o/{object}/acl', request_field=u'objectAccessControl', request_type_name=u'StorageObjectAccessControlsInsertRequest', response_type_name=u'ObjectAccessControl', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves ACL entries on the specified object.

      Args:
        request: (StorageObjectAccessControlsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ObjectAccessControls) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'storage.objectAccessControls.list', ordered_params=[u'bucket', u'object'], path_params=[u'bucket', u'object'], query_params=[u'generation', u'userProject'], relative_path=u'b/{bucket}/o/{object}/acl', request_field='', request_type_name=u'StorageObjectAccessControlsListRequest', response_type_name=u'ObjectAccessControls', supports_download=False)

    def Patch(self, request, global_params=None):
        """Patches an ACL entry on the specified object.

      Args:
        request: (StorageObjectAccessControlsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ObjectAccessControl) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PATCH', method_id=u'storage.objectAccessControls.patch', ordered_params=[u'bucket', u'object', u'entity'], path_params=[u'bucket', u'entity', u'object'], query_params=[u'generation', u'userProject'], relative_path=u'b/{bucket}/o/{object}/acl/{entity}', request_field=u'objectAccessControl', request_type_name=u'StorageObjectAccessControlsPatchRequest', response_type_name=u'ObjectAccessControl', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates an ACL entry on the specified object.

      Args:
        request: (StorageObjectAccessControlsUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ObjectAccessControl) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PUT', method_id=u'storage.objectAccessControls.update', ordered_params=[u'bucket', u'object', u'entity'], path_params=[u'bucket', u'entity', u'object'], query_params=[u'generation', u'userProject'], relative_path=u'b/{bucket}/o/{object}/acl/{entity}', request_field=u'objectAccessControl', request_type_name=u'StorageObjectAccessControlsUpdateRequest', response_type_name=u'ObjectAccessControl', supports_download=False)
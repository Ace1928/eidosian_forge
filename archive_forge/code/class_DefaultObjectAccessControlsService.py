import platform
import sys
from apitools.base.py import base_api
import gslib
from gslib.metrics import MetricsCollector
from gslib.third_party.storage_apitools import storage_v1_messages as messages
class DefaultObjectAccessControlsService(base_api.BaseApiService):
    """Service class for the defaultObjectAccessControls resource."""
    _NAME = u'defaultObjectAccessControls'

    def __init__(self, client):
        super(StorageV1.DefaultObjectAccessControlsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Permanently deletes the default object ACL entry for the specified entity on the specified bucket.

      Args:
        request: (StorageDefaultObjectAccessControlsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (StorageDefaultObjectAccessControlsDeleteResponse) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method=u'DELETE', method_id=u'storage.defaultObjectAccessControls.delete', ordered_params=[u'bucket', u'entity'], path_params=[u'bucket', u'entity'], query_params=[u'userProject'], relative_path=u'b/{bucket}/defaultObjectAcl/{entity}', request_field='', request_type_name=u'StorageDefaultObjectAccessControlsDeleteRequest', response_type_name=u'StorageDefaultObjectAccessControlsDeleteResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the default object ACL entry for the specified entity on the specified bucket.

      Args:
        request: (StorageDefaultObjectAccessControlsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ObjectAccessControl) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'storage.defaultObjectAccessControls.get', ordered_params=[u'bucket', u'entity'], path_params=[u'bucket', u'entity'], query_params=[u'userProject'], relative_path=u'b/{bucket}/defaultObjectAcl/{entity}', request_field='', request_type_name=u'StorageDefaultObjectAccessControlsGetRequest', response_type_name=u'ObjectAccessControl', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a new default object ACL entry on the specified bucket.

      Args:
        request: (StorageDefaultObjectAccessControlsInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ObjectAccessControl) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'storage.defaultObjectAccessControls.insert', ordered_params=[u'bucket'], path_params=[u'bucket'], query_params=[u'userProject'], relative_path=u'b/{bucket}/defaultObjectAcl', request_field=u'objectAccessControl', request_type_name=u'StorageDefaultObjectAccessControlsInsertRequest', response_type_name=u'ObjectAccessControl', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves default object ACL entries on the specified bucket.

      Args:
        request: (StorageDefaultObjectAccessControlsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ObjectAccessControls) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'storage.defaultObjectAccessControls.list', ordered_params=[u'bucket'], path_params=[u'bucket'], query_params=[u'ifMetagenerationMatch', u'ifMetagenerationNotMatch', u'userProject'], relative_path=u'b/{bucket}/defaultObjectAcl', request_field='', request_type_name=u'StorageDefaultObjectAccessControlsListRequest', response_type_name=u'ObjectAccessControls', supports_download=False)

    def Patch(self, request, global_params=None):
        """Patches a default object ACL entry on the specified bucket.

      Args:
        request: (StorageDefaultObjectAccessControlsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ObjectAccessControl) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PATCH', method_id=u'storage.defaultObjectAccessControls.patch', ordered_params=[u'bucket', u'entity'], path_params=[u'bucket', u'entity'], query_params=[u'userProject'], relative_path=u'b/{bucket}/defaultObjectAcl/{entity}', request_field=u'objectAccessControl', request_type_name=u'StorageDefaultObjectAccessControlsPatchRequest', response_type_name=u'ObjectAccessControl', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates a default object ACL entry on the specified bucket.

      Args:
        request: (StorageDefaultObjectAccessControlsUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ObjectAccessControl) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PUT', method_id=u'storage.defaultObjectAccessControls.update', ordered_params=[u'bucket', u'entity'], path_params=[u'bucket', u'entity'], query_params=[u'userProject'], relative_path=u'b/{bucket}/defaultObjectAcl/{entity}', request_field=u'objectAccessControl', request_type_name=u'StorageDefaultObjectAccessControlsUpdateRequest', response_type_name=u'ObjectAccessControl', supports_download=False)
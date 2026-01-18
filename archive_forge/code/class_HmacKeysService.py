import platform
import sys
from apitools.base.py import base_api
import gslib
from gslib.metrics import MetricsCollector
from gslib.third_party.storage_apitools import storage_v1_messages as messages
class HmacKeysService(base_api.BaseApiService):
    """Service class for the project_hmacKeys resource."""
    _NAME = u'projects_hmacKeys'

    def Create(self, request, global_params=None):
        """Creates HMAC key for a service account.

      Args:
        request: (StorageProjectsHmacKeysCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (HmacKey) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'storage.projects.hmacKeys.create', ordered_params=[u'serviceAccountEmail', u'projectId'], path_params=[u'projectId'], query_params=[u'serviceAccountEmail'], relative_path=u'projects/{projectId}/hmacKeys', request_field='', request_type_name=u'StorageProjectsHmacKeysCreateRequest', response_type_name=u'HmacKey', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an HMAC key.

      Args:
        request: (StorageProjectsHmacKeysDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (HmacKeyDeleteResponse) The empty response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method=u'DELETE', method_id=u'storage.projects.hmacKeys.delete', ordered_params=[u'accessId', u'projectId'], path_params=[u'accessId', u'projectId'], query_params=[], relative_path=u'projects/{projectId}/hmacKeys/{accessId}', request_field='', request_type_name=u'StorageProjectsHmacKeysDeleteRequest', response_type_name=u'HmacKeysDeleteResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves an HMAC key's metadata

      Args:
        request: (StorageProjectsHmacKeysGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (HmacKeyMetadata) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'storage.projects.hmacKeys.get', ordered_params=[u'accessId', u'projectId'], path_params=[u'accessId', u'projectId'], relative_path=u'projects/{projectId}/hmacKeys/{accessId}', request_field='', request_type_name=u'StorageProjectsHmacKeysGetRequest', response_type_name=u'HmacKeyMetadata', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of HMAC keys matching the criteria.

      Args:
        request: (StorageProjectsHmacKeysListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (HmacKeyMetadataList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'storage.projects.hmacKeys.get', ordered_params=[u'projectId', u'serviceAccountEmail', u'showDeletedKeys', u'maxResults', u'pageToken'], path_params=[u'projectId'], query_params=[u'serviceAccountEmail', u'showDeletedKeys', u'maxResults', u'pageToken'], relative_path=u'projects/{projectId}/hmacKeys', request_field='', request_type_name=u'StorageProjectsHmacKeysListRequest', response_type_name=u'HmacKeyMetadataList', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates the state of an HMAC key.

      Args:
        request: (StorageProjectsHmacKeysUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments

      Returns:
        (HmacKeyMetadata) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PUT', method_id=u'storage.projects.hmacKeys.update', ordered_params=[u'accessId', u'projectId'], path_params=[u'accessId', u'projectId'], query_params=[], relative_path=u'projects/{projectId}/hmacKeys/{accessId}', request_field='resource', request_type_name=u'StorageProjectsHmacKeysUpdateRequest', response_type_name=u'HmacKeyMetadata', supports_download=False)
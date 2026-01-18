from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.storage.v1 import storage_v1_messages as messages
class AnywhereCachesService(base_api.BaseApiService):
    """Service class for the anywhereCaches resource."""
    _NAME = 'anywhereCaches'

    def __init__(self, client):
        super(StorageV1.AnywhereCachesService, self).__init__(client)
        self._upload_configs = {}

    def Disable(self, request, global_params=None):
        """Disables an Anywhere Cache instance.

      Args:
        request: (StorageAnywhereCachesDisableRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AnywhereCache) The response message.
      """
        config = self.GetMethodConfig('Disable')
        return self._RunMethod(config, request, global_params=global_params)
    Disable.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='storage.anywhereCaches.disable', ordered_params=['bucket', 'anywhereCacheId'], path_params=['anywhereCacheId', 'bucket'], query_params=[], relative_path='b/{bucket}/anywhereCaches/{anywhereCacheId}/disable', request_field='', request_type_name='StorageAnywhereCachesDisableRequest', response_type_name='AnywhereCache', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the metadata of an Anywhere Cache instance.

      Args:
        request: (StorageAnywhereCachesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AnywhereCache) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='storage.anywhereCaches.get', ordered_params=['bucket', 'anywhereCacheId'], path_params=['anywhereCacheId', 'bucket'], query_params=[], relative_path='b/{bucket}/anywhereCaches/{anywhereCacheId}', request_field='', request_type_name='StorageAnywhereCachesGetRequest', response_type_name='AnywhereCache', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates an Anywhere Cache instance.

      Args:
        request: (AnywhereCache) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='storage.anywhereCaches.insert', ordered_params=['bucket'], path_params=['bucket'], query_params=[], relative_path='b/{bucket}/anywhereCaches', request_field='<request>', request_type_name='AnywhereCache', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def List(self, request, global_params=None):
        """Returns a list of Anywhere Cache instances of the bucket matching the criteria.

      Args:
        request: (StorageAnywhereCachesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AnywhereCaches) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='storage.anywhereCaches.list', ordered_params=['bucket'], path_params=['bucket'], query_params=['pageSize', 'pageToken'], relative_path='b/{bucket}/anywhereCaches', request_field='', request_type_name='StorageAnywhereCachesListRequest', response_type_name='AnywhereCaches', supports_download=False)

    def Pause(self, request, global_params=None):
        """Pauses an Anywhere Cache instance.

      Args:
        request: (StorageAnywhereCachesPauseRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AnywhereCache) The response message.
      """
        config = self.GetMethodConfig('Pause')
        return self._RunMethod(config, request, global_params=global_params)
    Pause.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='storage.anywhereCaches.pause', ordered_params=['bucket', 'anywhereCacheId'], path_params=['anywhereCacheId', 'bucket'], query_params=[], relative_path='b/{bucket}/anywhereCaches/{anywhereCacheId}/pause', request_field='', request_type_name='StorageAnywhereCachesPauseRequest', response_type_name='AnywhereCache', supports_download=False)

    def Resume(self, request, global_params=None):
        """Resumes a paused or disabled Anywhere Cache instance.

      Args:
        request: (StorageAnywhereCachesResumeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AnywhereCache) The response message.
      """
        config = self.GetMethodConfig('Resume')
        return self._RunMethod(config, request, global_params=global_params)
    Resume.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='storage.anywhereCaches.resume', ordered_params=['bucket', 'anywhereCacheId'], path_params=['anywhereCacheId', 'bucket'], query_params=[], relative_path='b/{bucket}/anywhereCaches/{anywhereCacheId}/resume', request_field='', request_type_name='StorageAnywhereCachesResumeRequest', response_type_name='AnywhereCache', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates the config(ttl and admissionPolicy) of an Anywhere Cache instance.

      Args:
        request: (AnywhereCache) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='storage.anywhereCaches.update', ordered_params=['bucket', 'anywhereCacheId'], path_params=['anywhereCacheId', 'bucket'], query_params=[], relative_path='b/{bucket}/anywhereCaches/{anywhereCacheId}', request_field='<request>', request_type_name='AnywhereCache', response_type_name='GoogleLongrunningOperation', supports_download=False)
import platform
import sys
from apitools.base.py import base_api
import gslib
from gslib.metrics import MetricsCollector
from gslib.third_party.storage_apitools import storage_v1_messages as messages
class ObjectsService(base_api.BaseApiService):
    """Service class for the objects resource."""
    _NAME = u'objects'

    def __init__(self, client):
        super(StorageV1.ObjectsService, self).__init__(client)
        self._upload_configs = {'Insert': base_api.ApiUploadInfo(accept=['*/*'], max_size=None, resumable_multipart=True, resumable_path=u'/resumable/upload/storage/' + self._client._version + '/b/{bucket}/o', simple_multipart=True, simple_path=u'/upload/storage/' + self._client._version + '/b/{bucket}/o')}

    def Compose(self, request, global_params=None):
        """Concatenates a list of existing objects into a new object in the same bucket.

      Args:
        request: (StorageObjectsComposeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Object) The response message.
      """
        config = self.GetMethodConfig('Compose')
        return self._RunMethod(config, request, global_params=global_params)
    Compose.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'storage.objects.compose', ordered_params=[u'destinationBucket', u'destinationObject'], path_params=[u'destinationBucket', u'destinationObject'], query_params=[u'destinationPredefinedAcl', u'ifGenerationMatch', u'ifMetagenerationMatch', u'kmsKeyName', u'userProject'], relative_path=u'b/{destinationBucket}/o/{destinationObject}/compose', request_field=u'composeRequest', request_type_name=u'StorageObjectsComposeRequest', response_type_name=u'Object', supports_download=False)

    def Copy(self, request, global_params=None):
        """Copies a source object to a destination object. Optionally overrides metadata.

      Args:
        request: (StorageObjectsCopyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Object) The response message.
      """
        config = self.GetMethodConfig('Copy')
        return self._RunMethod(config, request, global_params=global_params)
    Copy.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'storage.objects.copy', ordered_params=[u'sourceBucket', u'sourceObject', u'destinationBucket', u'destinationObject'], path_params=[u'destinationBucket', u'destinationObject', u'sourceBucket', u'sourceObject'], query_params=[u'destinationPredefinedAcl', u'ifGenerationMatch', u'ifGenerationNotMatch', u'ifMetagenerationMatch', u'ifMetagenerationNotMatch', u'ifSourceGenerationMatch', u'ifSourceGenerationNotMatch', u'ifSourceMetagenerationMatch', u'ifSourceMetagenerationNotMatch', u'projection', u'sourceGeneration', u'userProject'], relative_path=u'b/{sourceBucket}/o/{sourceObject}/copyTo/b/{destinationBucket}/o/{destinationObject}', request_field=u'object', request_type_name=u'StorageObjectsCopyRequest', response_type_name=u'Object', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an object and its metadata. Deletions are permanent if versioning is not enabled for the bucket, or if the generation parameter is used.

      Args:
        request: (StorageObjectsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (StorageObjectsDeleteResponse) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method=u'DELETE', method_id=u'storage.objects.delete', ordered_params=[u'bucket', u'object'], path_params=[u'bucket', u'object'], query_params=[u'generation', u'ifGenerationMatch', u'ifGenerationNotMatch', u'ifMetagenerationMatch', u'ifMetagenerationNotMatch', u'userProject'], relative_path=u'b/{bucket}/o/{object}', request_field='', request_type_name=u'StorageObjectsDeleteRequest', response_type_name=u'StorageObjectsDeleteResponse', supports_download=False)

    def Get(self, request, global_params=None, download=None):
        """Retrieves an object or its metadata.

      Args:
        request: (StorageObjectsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
        download: (Download, default: None) If present, download
            data from the request via this stream.
      Returns:
        (Object) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params, download=download)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'storage.objects.get', ordered_params=[u'bucket', u'object'], path_params=[u'bucket', u'object'], query_params=[u'generation', u'ifGenerationMatch', u'ifGenerationNotMatch', u'ifMetagenerationMatch', u'ifMetagenerationNotMatch', u'projection', u'userProject'], relative_path=u'b/{bucket}/o/{object}', request_field='', request_type_name=u'StorageObjectsGetRequest', response_type_name=u'Object', supports_download=True)

    def GetIamPolicy(self, request, global_params=None):
        """Returns an IAM policy for the specified object.

      Args:
        request: (StorageObjectsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'storage.objects.getIamPolicy', ordered_params=[u'bucket', u'object'], path_params=[u'bucket', u'object'], query_params=[u'generation', u'userProject'], relative_path=u'b/{bucket}/o/{object}/iam', request_field='', request_type_name=u'StorageObjectsGetIamPolicyRequest', response_type_name=u'Policy', supports_download=False)

    def Insert(self, request, global_params=None, upload=None):
        """Stores a new object and metadata.

      Args:
        request: (StorageObjectsInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
        upload: (Upload, default: None) If present, upload
            this stream with the request.
      Returns:
        (Object) The response message.
      """
        config = self.GetMethodConfig('Insert')
        upload_config = self.GetUploadConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params, upload=upload, upload_config=upload_config)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'storage.objects.insert', ordered_params=[u'bucket'], path_params=[u'bucket'], query_params=[u'contentEncoding', u'ifGenerationMatch', u'ifGenerationNotMatch', u'ifMetagenerationMatch', u'ifMetagenerationNotMatch', u'kmsKeyName', u'name', u'predefinedAcl', u'projection', u'userProject'], relative_path=u'b/{bucket}/o', request_field=u'object', request_type_name=u'StorageObjectsInsertRequest', response_type_name=u'Object', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves a list of objects matching the criteria.

      Args:
        request: (StorageObjectsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Objects) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'storage.objects.list', ordered_params=[u'bucket'], path_params=[u'bucket'], query_params=[u'delimiter', u'includeTrailingDelimiter', u'maxResults', u'pageToken', u'prefix', u'projection', u'userProject', u'versions'], relative_path=u'b/{bucket}/o', request_field='', request_type_name=u'StorageObjectsListRequest', response_type_name=u'Objects', supports_download=False)

    def Patch(self, request, global_params=None):
        """Patches an object's metadata.

      Args:
        request: (StorageObjectsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Object) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PATCH', method_id=u'storage.objects.patch', ordered_params=[u'bucket', u'object'], path_params=[u'bucket', u'object'], query_params=[u'generation', u'ifGenerationMatch', u'ifGenerationNotMatch', u'ifMetagenerationMatch', u'ifMetagenerationNotMatch', u'predefinedAcl', u'projection', u'userProject'], relative_path=u'b/{bucket}/o/{object}', request_field=u'objectResource', request_type_name=u'StorageObjectsPatchRequest', response_type_name=u'Object', supports_download=False)

    def Rewrite(self, request, global_params=None):
        """Rewrites a source object to a destination object. Optionally overrides metadata.

      Args:
        request: (StorageObjectsRewriteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RewriteResponse) The response message.
      """
        config = self.GetMethodConfig('Rewrite')
        return self._RunMethod(config, request, global_params=global_params)
    Rewrite.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'storage.objects.rewrite', ordered_params=[u'sourceBucket', u'sourceObject', u'destinationBucket', u'destinationObject'], path_params=[u'destinationBucket', u'destinationObject', u'sourceBucket', u'sourceObject'], query_params=[u'destinationKmsKeyName', u'destinationPredefinedAcl', u'ifGenerationMatch', u'ifGenerationNotMatch', u'ifMetagenerationMatch', u'ifMetagenerationNotMatch', u'ifSourceGenerationMatch', u'ifSourceGenerationNotMatch', u'ifSourceMetagenerationMatch', u'ifSourceMetagenerationNotMatch', u'maxBytesRewrittenPerCall', u'projection', u'rewriteToken', u'sourceGeneration', u'userProject'], relative_path=u'b/{sourceBucket}/o/{sourceObject}/rewriteTo/b/{destinationBucket}/o/{destinationObject}', request_field=u'object', request_type_name=u'StorageObjectsRewriteRequest', response_type_name=u'RewriteResponse', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Updates an IAM policy for the specified object.

      Args:
        request: (StorageObjectsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PUT', method_id=u'storage.objects.setIamPolicy', ordered_params=[u'bucket', u'object'], path_params=[u'bucket', u'object'], query_params=[u'generation', u'userProject'], relative_path=u'b/{bucket}/o/{object}/iam', request_field=u'policy', request_type_name=u'StorageObjectsSetIamPolicyRequest', response_type_name=u'Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Tests a set of permissions on the given object to see which, if any, are held by the caller.

      Args:
        request: (StorageObjectsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'storage.objects.testIamPermissions', ordered_params=[u'bucket', u'object', u'permissions'], path_params=[u'bucket', u'object'], query_params=[u'generation', u'permissions', u'userProject'], relative_path=u'b/{bucket}/o/{object}/iam/testPermissions', request_field='', request_type_name=u'StorageObjectsTestIamPermissionsRequest', response_type_name=u'TestIamPermissionsResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates an object's metadata.

      Args:
        request: (StorageObjectsUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Object) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(http_method=u'PUT', method_id=u'storage.objects.update', ordered_params=[u'bucket', u'object'], path_params=[u'bucket', u'object'], query_params=[u'generation', u'ifGenerationMatch', u'ifGenerationNotMatch', u'ifMetagenerationMatch', u'ifMetagenerationNotMatch', u'predefinedAcl', u'projection', u'userProject'], relative_path=u'b/{bucket}/o/{object}', request_field=u'objectResource', request_type_name=u'StorageObjectsUpdateRequest', response_type_name=u'Object', supports_download=False)

    def WatchAll(self, request, global_params=None):
        """Watch for changes on all objects in a bucket.

      Args:
        request: (StorageObjectsWatchAllRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Channel) The response message.
      """
        config = self.GetMethodConfig('WatchAll')
        return self._RunMethod(config, request, global_params=global_params)
    WatchAll.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'storage.objects.watchAll', ordered_params=[u'bucket'], path_params=[u'bucket'], query_params=[u'delimiter', u'includeTrailingDelimiter', u'maxResults', u'pageToken', u'prefix', u'projection', u'userProject', u'versions'], relative_path=u'b/{bucket}/o/watch', request_field=u'channel', request_type_name=u'StorageObjectsWatchAllRequest', response_type_name=u'Channel', supports_download=False)
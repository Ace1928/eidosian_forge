from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.logging.v2 import logging_v2_messages as messages
class LocationsBucketsService(base_api.BaseApiService):
    """Service class for the locations_buckets resource."""
    _NAME = 'locations_buckets'

    def __init__(self, client):
        super(LoggingV2.LocationsBucketsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a log bucket that can be used to store log entries. After a bucket has been created, the bucket's location cannot be changed.

      Args:
        request: (LoggingLocationsBucketsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LogBucket) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/{v2Id}/{v2Id1}/locations/{locationsId}/buckets', http_method='POST', method_id='logging.locations.buckets.create', ordered_params=['parent'], path_params=['parent'], query_params=['bucketId'], relative_path='v2/{+parent}/buckets', request_field='logBucket', request_type_name='LoggingLocationsBucketsCreateRequest', response_type_name='LogBucket', supports_download=False)

    def CreateAsync(self, request, global_params=None):
        """Creates a log bucket asynchronously that can be used to store log entries.After a bucket has been created, the bucket's location cannot be changed.

      Args:
        request: (LoggingLocationsBucketsCreateAsyncRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('CreateAsync')
        return self._RunMethod(config, request, global_params=global_params)
    CreateAsync.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/{v2Id}/{v2Id1}/locations/{locationsId}/buckets:createAsync', http_method='POST', method_id='logging.locations.buckets.createAsync', ordered_params=['parent'], path_params=['parent'], query_params=['bucketId'], relative_path='v2/{+parent}/buckets:createAsync', request_field='logBucket', request_type_name='LoggingLocationsBucketsCreateAsyncRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a log bucket.Changes the bucket's lifecycle_state to the DELETE_REQUESTED state. After 7 days, the bucket will be purged and all log entries in the bucket will be permanently deleted.

      Args:
        request: (LoggingLocationsBucketsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/{v2Id}/{v2Id1}/locations/{locationsId}/buckets/{bucketsId}', http_method='DELETE', method_id='logging.locations.buckets.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='LoggingLocationsBucketsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a log bucket.

      Args:
        request: (LoggingLocationsBucketsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LogBucket) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/{v2Id}/{v2Id1}/locations/{locationsId}/buckets/{bucketsId}', http_method='GET', method_id='logging.locations.buckets.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='LoggingLocationsBucketsGetRequest', response_type_name='LogBucket', supports_download=False)

    def List(self, request, global_params=None):
        """Lists log buckets.

      Args:
        request: (LoggingLocationsBucketsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListBucketsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/{v2Id}/{v2Id1}/locations/{locationsId}/buckets', http_method='GET', method_id='logging.locations.buckets.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2/{+parent}/buckets', request_field='', request_type_name='LoggingLocationsBucketsListRequest', response_type_name='ListBucketsResponse', supports_download=False)

    def Move(self, request, global_params=None):
        """Moves a bucket from one location to another location. This method creates a new bucket at the new location with an ACTIVE state. The bucket at the old location will remain available with an ARCHIVED state such that it is queryable but can no longer be used as a sink destination. All corresponding sinks are updated to point to the new bucket. Currently, the contents of the old bucket are not copied to the new one. In order to be movable, a bucket must satisfy the following restrictions: It's a _Default or _Required bucket. It has a location of "global". It has a non-project parent when it's a _Default bucket.

      Args:
        request: (MoveBucketRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Move')
        return self._RunMethod(config, request, global_params=global_params)
    Move.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/{v2Id}/{v2Id1}/locations/{locationsId}/buckets/{bucketsId}:move', http_method='POST', method_id='logging.locations.buckets.move', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:move', request_field='<request>', request_type_name='MoveBucketRequest', response_type_name='Operation', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a log bucket.If the bucket has a lifecycle_state of DELETE_REQUESTED, then FAILED_PRECONDITION will be returned.After a bucket has been created, the bucket's location cannot be changed.

      Args:
        request: (LoggingLocationsBucketsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LogBucket) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/{v2Id}/{v2Id1}/locations/{locationsId}/buckets/{bucketsId}', http_method='PATCH', method_id='logging.locations.buckets.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}', request_field='logBucket', request_type_name='LoggingLocationsBucketsPatchRequest', response_type_name='LogBucket', supports_download=False)

    def Undelete(self, request, global_params=None):
        """Undeletes a log bucket. A bucket that has been deleted can be undeleted within the grace period of 7 days.

      Args:
        request: (LoggingLocationsBucketsUndeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Undelete')
        return self._RunMethod(config, request, global_params=global_params)
    Undelete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/{v2Id}/{v2Id1}/locations/{locationsId}/buckets/{bucketsId}:undelete', http_method='POST', method_id='logging.locations.buckets.undelete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:undelete', request_field='undeleteBucketRequest', request_type_name='LoggingLocationsBucketsUndeleteRequest', response_type_name='Empty', supports_download=False)

    def UpdateAsync(self, request, global_params=None):
        """Updates a log bucket asynchronously.If the bucket has a lifecycle_state of DELETE_REQUESTED, then FAILED_PRECONDITION will be returned.After a bucket has been created, the bucket's location cannot be changed.

      Args:
        request: (LoggingLocationsBucketsUpdateAsyncRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('UpdateAsync')
        return self._RunMethod(config, request, global_params=global_params)
    UpdateAsync.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/{v2Id}/{v2Id1}/locations/{locationsId}/buckets/{bucketsId}:updateAsync', http_method='POST', method_id='logging.locations.buckets.updateAsync', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}:updateAsync', request_field='logBucket', request_type_name='LoggingLocationsBucketsUpdateAsyncRequest', response_type_name='Operation', supports_download=False)
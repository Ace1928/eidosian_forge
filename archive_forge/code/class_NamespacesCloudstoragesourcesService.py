from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.anthosevents.v1 import anthosevents_v1_messages as messages
class NamespacesCloudstoragesourcesService(base_api.BaseApiService):
    """Service class for the namespaces_cloudstoragesources resource."""
    _NAME = 'namespaces_cloudstoragesources'

    def __init__(self, client):
        super(AnthoseventsV1.NamespacesCloudstoragesourcesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new cloudstoragesource.

      Args:
        request: (AnthoseventsNamespacesCloudstoragesourcesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CloudStorageSource) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/events.cloud.google.com/v1/namespaces/{namespacesId}/cloudstoragesources', http_method='POST', method_id='anthosevents.namespaces.cloudstoragesources.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='apis/events.cloud.google.com/v1/{+parent}/cloudstoragesources', request_field='cloudStorageSource', request_type_name='AnthoseventsNamespacesCloudstoragesourcesCreateRequest', response_type_name='CloudStorageSource', supports_download=False)

    def Delete(self, request, global_params=None):
        """Rpc to delete a cloudstoragesource.

      Args:
        request: (AnthoseventsNamespacesCloudstoragesourcesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/events.cloud.google.com/v1/namespaces/{namespacesId}/cloudstoragesources/{cloudstoragesourcesId}', http_method='DELETE', method_id='anthosevents.namespaces.cloudstoragesources.delete', ordered_params=['name'], path_params=['name'], query_params=['apiVersion', 'kind', 'propagationPolicy'], relative_path='apis/events.cloud.google.com/v1/{+name}', request_field='', request_type_name='AnthoseventsNamespacesCloudstoragesourcesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Rpc to get information about a cloudstoragesource.

      Args:
        request: (AnthoseventsNamespacesCloudstoragesourcesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CloudStorageSource) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/events.cloud.google.com/v1/namespaces/{namespacesId}/cloudstoragesources/{cloudstoragesourcesId}', http_method='GET', method_id='anthosevents.namespaces.cloudstoragesources.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='apis/events.cloud.google.com/v1/{+name}', request_field='', request_type_name='AnthoseventsNamespacesCloudstoragesourcesGetRequest', response_type_name='CloudStorageSource', supports_download=False)

    def List(self, request, global_params=None):
        """Rpc to list cloudstoragesources.

      Args:
        request: (AnthoseventsNamespacesCloudstoragesourcesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListCloudStorageSourcesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/events.cloud.google.com/v1/namespaces/{namespacesId}/cloudstoragesources', http_method='GET', method_id='anthosevents.namespaces.cloudstoragesources.list', ordered_params=['parent'], path_params=['parent'], query_params=['continue_', 'fieldSelector', 'includeUninitialized', 'labelSelector', 'limit', 'resourceVersion', 'watch'], relative_path='apis/events.cloud.google.com/v1/{+parent}/cloudstoragesources', request_field='', request_type_name='AnthoseventsNamespacesCloudstoragesourcesListRequest', response_type_name='ListCloudStorageSourcesResponse', supports_download=False)

    def ReplaceCloudStorageSource(self, request, global_params=None):
        """Rpc to replace a cloudstoragesource. Only the spec and metadata labels and annotations are modifiable. After the Update request, Cloud Run will work to make the 'status' match the requested 'spec'. May provide metadata.resourceVersion to enforce update from last read for optimistic concurrency control.

      Args:
        request: (AnthoseventsNamespacesCloudstoragesourcesReplaceCloudStorageSourceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CloudStorageSource) The response message.
      """
        config = self.GetMethodConfig('ReplaceCloudStorageSource')
        return self._RunMethod(config, request, global_params=global_params)
    ReplaceCloudStorageSource.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/events.cloud.google.com/v1/namespaces/{namespacesId}/cloudstoragesources/{cloudstoragesourcesId}', http_method='PUT', method_id='anthosevents.namespaces.cloudstoragesources.replaceCloudStorageSource', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='apis/events.cloud.google.com/v1/{+name}', request_field='cloudStorageSource', request_type_name='AnthoseventsNamespacesCloudstoragesourcesReplaceCloudStorageSourceRequest', response_type_name='CloudStorageSource', supports_download=False)
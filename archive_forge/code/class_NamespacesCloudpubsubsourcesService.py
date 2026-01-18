from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.anthosevents.v1 import anthosevents_v1_messages as messages
class NamespacesCloudpubsubsourcesService(base_api.BaseApiService):
    """Service class for the namespaces_cloudpubsubsources resource."""
    _NAME = 'namespaces_cloudpubsubsources'

    def __init__(self, client):
        super(AnthoseventsV1.NamespacesCloudpubsubsourcesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new cloudpubsubsource.

      Args:
        request: (AnthoseventsNamespacesCloudpubsubsourcesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CloudPubSubSource) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/events.cloud.google.com/v1/namespaces/{namespacesId}/cloudpubsubsources', http_method='POST', method_id='anthosevents.namespaces.cloudpubsubsources.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='apis/events.cloud.google.com/v1/{+parent}/cloudpubsubsources', request_field='cloudPubSubSource', request_type_name='AnthoseventsNamespacesCloudpubsubsourcesCreateRequest', response_type_name='CloudPubSubSource', supports_download=False)

    def Delete(self, request, global_params=None):
        """Rpc to delete a cloudpubsubsource.

      Args:
        request: (AnthoseventsNamespacesCloudpubsubsourcesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/events.cloud.google.com/v1/namespaces/{namespacesId}/cloudpubsubsources/{cloudpubsubsourcesId}', http_method='DELETE', method_id='anthosevents.namespaces.cloudpubsubsources.delete', ordered_params=['name'], path_params=['name'], query_params=['apiVersion', 'kind', 'propagationPolicy'], relative_path='apis/events.cloud.google.com/v1/{+name}', request_field='', request_type_name='AnthoseventsNamespacesCloudpubsubsourcesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Rpc to get information about a cloudpubsubsource.

      Args:
        request: (AnthoseventsNamespacesCloudpubsubsourcesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CloudPubSubSource) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/events.cloud.google.com/v1/namespaces/{namespacesId}/cloudpubsubsources/{cloudpubsubsourcesId}', http_method='GET', method_id='anthosevents.namespaces.cloudpubsubsources.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='apis/events.cloud.google.com/v1/{+name}', request_field='', request_type_name='AnthoseventsNamespacesCloudpubsubsourcesGetRequest', response_type_name='CloudPubSubSource', supports_download=False)

    def List(self, request, global_params=None):
        """Rpc to list cloudpubsubsources.

      Args:
        request: (AnthoseventsNamespacesCloudpubsubsourcesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListCloudPubSubSourcesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/events.cloud.google.com/v1/namespaces/{namespacesId}/cloudpubsubsources', http_method='GET', method_id='anthosevents.namespaces.cloudpubsubsources.list', ordered_params=['parent'], path_params=['parent'], query_params=['continue_', 'fieldSelector', 'includeUninitialized', 'labelSelector', 'limit', 'resourceVersion', 'watch'], relative_path='apis/events.cloud.google.com/v1/{+parent}/cloudpubsubsources', request_field='', request_type_name='AnthoseventsNamespacesCloudpubsubsourcesListRequest', response_type_name='ListCloudPubSubSourcesResponse', supports_download=False)

    def ReplaceCloudPubSubSource(self, request, global_params=None):
        """Rpc to replace a cloudpubsubsource. Only the spec and metadata labels and annotations are modifiable. After the Update request, Cloud Run will work to make the 'status' match the requested 'spec'. May provide metadata.resourceVersion to enforce update from last read for optimistic concurrency control.

      Args:
        request: (AnthoseventsNamespacesCloudpubsubsourcesReplaceCloudPubSubSourceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CloudPubSubSource) The response message.
      """
        config = self.GetMethodConfig('ReplaceCloudPubSubSource')
        return self._RunMethod(config, request, global_params=global_params)
    ReplaceCloudPubSubSource.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/events.cloud.google.com/v1/namespaces/{namespacesId}/cloudpubsubsources/{cloudpubsubsourcesId}', http_method='PUT', method_id='anthosevents.namespaces.cloudpubsubsources.replaceCloudPubSubSource', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='apis/events.cloud.google.com/v1/{+name}', request_field='cloudPubSubSource', request_type_name='AnthoseventsNamespacesCloudpubsubsourcesReplaceCloudPubSubSourceRequest', response_type_name='CloudPubSubSource', supports_download=False)
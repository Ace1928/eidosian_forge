from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.anthosevents.v1 import anthosevents_v1_messages as messages
class NamespacesTriggersService(base_api.BaseApiService):
    """Service class for the namespaces_triggers resource."""
    _NAME = 'namespaces_triggers'

    def __init__(self, client):
        super(AnthoseventsV1.NamespacesTriggersService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new trigger.

      Args:
        request: (AnthoseventsNamespacesTriggersCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Trigger) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/eventing.knative.dev/v1/namespaces/{namespacesId}/triggers', http_method='POST', method_id='anthosevents.namespaces.triggers.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='apis/eventing.knative.dev/v1/{+parent}/triggers', request_field='trigger', request_type_name='AnthoseventsNamespacesTriggersCreateRequest', response_type_name='Trigger', supports_download=False)

    def Delete(self, request, global_params=None):
        """Rpc to delete a trigger.

      Args:
        request: (AnthoseventsNamespacesTriggersDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/eventing.knative.dev/v1/namespaces/{namespacesId}/triggers/{triggersId}', http_method='DELETE', method_id='anthosevents.namespaces.triggers.delete', ordered_params=['name'], path_params=['name'], query_params=['apiVersion', 'kind', 'propagationPolicy'], relative_path='apis/eventing.knative.dev/v1/{+name}', request_field='', request_type_name='AnthoseventsNamespacesTriggersDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Rpc to get information about a trigger.

      Args:
        request: (AnthoseventsNamespacesTriggersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Trigger) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/eventing.knative.dev/v1/namespaces/{namespacesId}/triggers/{triggersId}', http_method='GET', method_id='anthosevents.namespaces.triggers.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='apis/eventing.knative.dev/v1/{+name}', request_field='', request_type_name='AnthoseventsNamespacesTriggersGetRequest', response_type_name='Trigger', supports_download=False)

    def List(self, request, global_params=None):
        """Rpc to list triggers.

      Args:
        request: (AnthoseventsNamespacesTriggersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListTriggersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/eventing.knative.dev/v1/namespaces/{namespacesId}/triggers', http_method='GET', method_id='anthosevents.namespaces.triggers.list', ordered_params=['parent'], path_params=['parent'], query_params=['continue_', 'fieldSelector', 'includeUninitialized', 'labelSelector', 'pageSize', 'resourceVersion', 'watch'], relative_path='apis/eventing.knative.dev/v1/{+parent}/triggers', request_field='', request_type_name='AnthoseventsNamespacesTriggersListRequest', response_type_name='ListTriggersResponse', supports_download=False)

    def ReplaceTrigger(self, request, global_params=None):
        """Rpc to replace a trigger. Only the spec and metadata labels and annotations are modifiable. After the Update request, Events for Cloud Run will work to make the 'status' match the requested 'spec'. May provide metadata.resourceVersion to enforce update from last read for optimistic concurrency control.

      Args:
        request: (AnthoseventsNamespacesTriggersReplaceTriggerRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Trigger) The response message.
      """
        config = self.GetMethodConfig('ReplaceTrigger')
        return self._RunMethod(config, request, global_params=global_params)
    ReplaceTrigger.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/eventing.knative.dev/v1/namespaces/{namespacesId}/triggers/{triggersId}', http_method='PUT', method_id='anthosevents.namespaces.triggers.replaceTrigger', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='apis/eventing.knative.dev/v1/{+name}', request_field='trigger', request_type_name='AnthoseventsNamespacesTriggersReplaceTriggerRequest', response_type_name='Trigger', supports_download=False)
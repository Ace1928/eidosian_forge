from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.alpha import compute_alpha_messages as messages
class ZoneQueuedResourcesService(base_api.BaseApiService):
    """Service class for the zoneQueuedResources resource."""
    _NAME = 'zoneQueuedResources'

    def __init__(self, client):
        super(ComputeAlpha.ZoneQueuedResourcesService, self).__init__(client)
        self._upload_configs = {}

    def AggregatedList(self, request, global_params=None):
        """Retrieves an aggregated list of all of the queued resources in a project across all zones. To prevent failure, Google recommends that you set the `returnPartialSuccess` parameter to `true`.

      Args:
        request: (ComputeZoneQueuedResourcesAggregatedListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (QueuedResourcesAggregatedList) The response message.
      """
        config = self.GetMethodConfig('AggregatedList')
        return self._RunMethod(config, request, global_params=global_params)
    AggregatedList.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.zoneQueuedResources.aggregatedList', ordered_params=['project'], path_params=['project'], query_params=['filter', 'includeAllScopes', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess', 'serviceProjectNumber'], relative_path='projects/{project}/aggregated/queuedResources', request_field='', request_type_name='ComputeZoneQueuedResourcesAggregatedListRequest', response_type_name='QueuedResourcesAggregatedList', supports_download=False)

    def Cancel(self, request, global_params=None):
        """Cancels a QueuedResource. Only a resource in ACCEPTED state may be cancelled.

      Args:
        request: (ComputeZoneQueuedResourcesCancelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Cancel')
        return self._RunMethod(config, request, global_params=global_params)
    Cancel.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.zoneQueuedResources.cancel', ordered_params=['project', 'zone', 'queuedResource'], path_params=['project', 'queuedResource', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/queuedResources/{queuedResource}/cancel', request_field='', request_type_name='ComputeZoneQueuedResourcesCancelRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a QueuedResource. For a QueuedResource in ACCEPTED state, call cancel on the resource before deleting, to make sure no VMs have been provisioned and may require cleaning up. For a QueuedResource in PROVISIONING state the request to delete is registered for execution following the provisioning.

      Args:
        request: (ComputeZoneQueuedResourcesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='compute.zoneQueuedResources.delete', ordered_params=['project', 'zone', 'queuedResource'], path_params=['project', 'queuedResource', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/queuedResources/{queuedResource}', request_field='', request_type_name='ComputeZoneQueuedResourcesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the specified QueuedResource resource.

      Args:
        request: (ComputeZoneQueuedResourcesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (QueuedResource) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.zoneQueuedResources.get', ordered_params=['project', 'zone', 'queuedResource'], path_params=['project', 'queuedResource', 'zone'], query_params=[], relative_path='projects/{project}/zones/{zone}/queuedResources/{queuedResource}', request_field='', request_type_name='ComputeZoneQueuedResourcesGetRequest', response_type_name='QueuedResource', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a QueuedResource.

      Args:
        request: (ComputeZoneQueuedResourcesInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='compute.zoneQueuedResources.insert', ordered_params=['project', 'zone'], path_params=['project', 'zone'], query_params=['requestId'], relative_path='projects/{project}/zones/{zone}/queuedResources', request_field='queuedResource', request_type_name='ComputeZoneQueuedResourcesInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Retrieves the list of QueuedResource resources.

      Args:
        request: (ComputeZoneQueuedResourcesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (QueuedResourceList) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='compute.zoneQueuedResources.list', ordered_params=['project', 'zone'], path_params=['project', 'zone'], query_params=['filter', 'maxResults', 'orderBy', 'pageToken', 'returnPartialSuccess'], relative_path='projects/{project}/zones/{zone}/queuedResources', request_field='', request_type_name='ComputeZoneQueuedResourcesListRequest', response_type_name='QueuedResourceList', supports_download=False)
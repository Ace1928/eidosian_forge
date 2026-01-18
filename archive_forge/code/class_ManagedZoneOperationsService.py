from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dns.v1alpha2 import dns_v1alpha2_messages as messages
class ManagedZoneOperationsService(base_api.BaseApiService):
    """Service class for the managedZoneOperations resource."""
    _NAME = 'managedZoneOperations'

    def __init__(self, client):
        super(DnsV1alpha2.ManagedZoneOperationsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Fetches the representation of an existing Operation.

      Args:
        request: (DnsManagedZoneOperationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='dns.managedZoneOperations.get', ordered_params=['project', 'managedZone', 'operation'], path_params=['managedZone', 'operation', 'project'], query_params=['clientOperationId'], relative_path='dns/v1alpha2/projects/{project}/managedZones/{managedZone}/operations/{operation}', request_field='', request_type_name='DnsManagedZoneOperationsGetRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Enumerates Operations for the given ManagedZone.

      Args:
        request: (DnsManagedZoneOperationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ManagedZoneOperationsListResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='dns.managedZoneOperations.list', ordered_params=['project', 'managedZone'], path_params=['managedZone', 'project'], query_params=['maxResults', 'pageToken', 'sortBy'], relative_path='dns/v1alpha2/projects/{project}/managedZones/{managedZone}/operations', request_field='', request_type_name='DnsManagedZoneOperationsListRequest', response_type_name='ManagedZoneOperationsListResponse', supports_download=False)
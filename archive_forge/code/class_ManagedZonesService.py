from __future__ import absolute_import
from apitools.base.py import base_api
from samples.dns_sample.dns_v1 import dns_v1_messages as messages
class ManagedZonesService(base_api.BaseApiService):
    """Service class for the managedZones resource."""
    _NAME = u'managedZones'

    def __init__(self, client):
        super(DnsV1.ManagedZonesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create a new ManagedZone.

      Args:
        request: (DnsManagedZonesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ManagedZone) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(http_method=u'POST', method_id=u'dns.managedZones.create', ordered_params=[u'project'], path_params=[u'project'], query_params=[], relative_path=u'projects/{project}/managedZones', request_field=u'managedZone', request_type_name=u'DnsManagedZonesCreateRequest', response_type_name=u'ManagedZone', supports_download=False)

    def Delete(self, request, global_params=None):
        """Delete a previously created ManagedZone.

      Args:
        request: (DnsManagedZonesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DnsManagedZonesDeleteResponse) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method=u'DELETE', method_id=u'dns.managedZones.delete', ordered_params=[u'project', u'managedZone'], path_params=[u'managedZone', u'project'], query_params=[], relative_path=u'projects/{project}/managedZones/{managedZone}', request_field='', request_type_name=u'DnsManagedZonesDeleteRequest', response_type_name=u'DnsManagedZonesDeleteResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Fetch the representation of an existing ManagedZone.

      Args:
        request: (DnsManagedZonesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ManagedZone) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'dns.managedZones.get', ordered_params=[u'project', u'managedZone'], path_params=[u'managedZone', u'project'], query_params=[], relative_path=u'projects/{project}/managedZones/{managedZone}', request_field='', request_type_name=u'DnsManagedZonesGetRequest', response_type_name=u'ManagedZone', supports_download=False)

    def List(self, request, global_params=None):
        """Enumerate ManagedZones that have been created but not yet deleted.

      Args:
        request: (DnsManagedZonesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ManagedZonesListResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method=u'GET', method_id=u'dns.managedZones.list', ordered_params=[u'project'], path_params=[u'project'], query_params=[u'dnsName', u'maxResults', u'pageToken'], relative_path=u'projects/{project}/managedZones', request_field='', request_type_name=u'DnsManagedZonesListRequest', response_type_name=u'ManagedZonesListResponse', supports_download=False)
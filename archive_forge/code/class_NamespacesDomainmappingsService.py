from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.run.v1 import run_v1_messages as messages
class NamespacesDomainmappingsService(base_api.BaseApiService):
    """Service class for the namespaces_domainmappings resource."""
    _NAME = 'namespaces_domainmappings'

    def __init__(self, client):
        super(RunV1.NamespacesDomainmappingsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create a new domain mapping.

      Args:
        request: (RunNamespacesDomainmappingsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DomainMapping) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/domains.cloudrun.com/v1/namespaces/{namespacesId}/domainmappings', http_method='POST', method_id='run.namespaces.domainmappings.create', ordered_params=['parent'], path_params=['parent'], query_params=['dryRun'], relative_path='apis/domains.cloudrun.com/v1/{+parent}/domainmappings', request_field='domainMapping', request_type_name='RunNamespacesDomainmappingsCreateRequest', response_type_name='DomainMapping', supports_download=False)

    def Delete(self, request, global_params=None):
        """Delete a domain mapping.

      Args:
        request: (RunNamespacesDomainmappingsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Status) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/domains.cloudrun.com/v1/namespaces/{namespacesId}/domainmappings/{domainmappingsId}', http_method='DELETE', method_id='run.namespaces.domainmappings.delete', ordered_params=['name'], path_params=['name'], query_params=['apiVersion', 'dryRun', 'kind', 'propagationPolicy'], relative_path='apis/domains.cloudrun.com/v1/{+name}', request_field='', request_type_name='RunNamespacesDomainmappingsDeleteRequest', response_type_name='Status', supports_download=False)

    def Get(self, request, global_params=None):
        """Get information about a domain mapping.

      Args:
        request: (RunNamespacesDomainmappingsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DomainMapping) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/domains.cloudrun.com/v1/namespaces/{namespacesId}/domainmappings/{domainmappingsId}', http_method='GET', method_id='run.namespaces.domainmappings.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='apis/domains.cloudrun.com/v1/{+name}', request_field='', request_type_name='RunNamespacesDomainmappingsGetRequest', response_type_name='DomainMapping', supports_download=False)

    def List(self, request, global_params=None):
        """List all domain mappings.

      Args:
        request: (RunNamespacesDomainmappingsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDomainMappingsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='apis/domains.cloudrun.com/v1/namespaces/{namespacesId}/domainmappings', http_method='GET', method_id='run.namespaces.domainmappings.list', ordered_params=['parent'], path_params=['parent'], query_params=['continue_', 'fieldSelector', 'includeUninitialized', 'labelSelector', 'limit', 'resourceVersion', 'watch'], relative_path='apis/domains.cloudrun.com/v1/{+parent}/domainmappings', request_field='', request_type_name='RunNamespacesDomainmappingsListRequest', response_type_name='ListDomainMappingsResponse', supports_download=False)
from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.appengine.v1beta import appengine_v1beta_messages as messages
class AppsDomainMappingsService(base_api.BaseApiService):
    """Service class for the apps_domainMappings resource."""
    _NAME = 'apps_domainMappings'

    def __init__(self, client):
        super(AppengineV1beta.AppsDomainMappingsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Maps a domain to an application. A user must be authorized to administer a domain in order to map it to an application. For a list of available authorized domains, see AuthorizedDomains.ListAuthorizedDomains.

      Args:
        request: (AppengineAppsDomainMappingsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/apps/{appsId}/domainMappings', http_method='POST', method_id='appengine.apps.domainMappings.create', ordered_params=['parent'], path_params=['parent'], query_params=['overrideStrategy'], relative_path='v1beta/{+parent}/domainMappings', request_field='domainMapping', request_type_name='AppengineAppsDomainMappingsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified domain mapping. A user must be authorized to administer the associated domain in order to delete a DomainMapping resource.

      Args:
        request: (AppengineAppsDomainMappingsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/apps/{appsId}/domainMappings/{domainMappingsId}', http_method='DELETE', method_id='appengine.apps.domainMappings.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='AppengineAppsDomainMappingsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the specified domain mapping.

      Args:
        request: (AppengineAppsDomainMappingsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DomainMapping) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/apps/{appsId}/domainMappings/{domainMappingsId}', http_method='GET', method_id='appengine.apps.domainMappings.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='AppengineAppsDomainMappingsGetRequest', response_type_name='DomainMapping', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the domain mappings on an application.

      Args:
        request: (AppengineAppsDomainMappingsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDomainMappingsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/apps/{appsId}/domainMappings', http_method='GET', method_id='appengine.apps.domainMappings.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1beta/{+parent}/domainMappings', request_field='', request_type_name='AppengineAppsDomainMappingsListRequest', response_type_name='ListDomainMappingsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the specified domain mapping. To map an SSL certificate to a domain mapping, update certificate_id to point to an AuthorizedCertificate resource. A user must be authorized to administer the associated domain in order to update a DomainMapping resource.

      Args:
        request: (AppengineAppsDomainMappingsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/apps/{appsId}/domainMappings/{domainMappingsId}', http_method='PATCH', method_id='appengine.apps.domainMappings.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1beta/{+name}', request_field='domainMapping', request_type_name='AppengineAppsDomainMappingsPatchRequest', response_type_name='Operation', supports_download=False)
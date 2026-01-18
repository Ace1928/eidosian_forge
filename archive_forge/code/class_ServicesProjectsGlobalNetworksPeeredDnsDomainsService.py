from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.servicenetworking.v1 import servicenetworking_v1_messages as messages
class ServicesProjectsGlobalNetworksPeeredDnsDomainsService(base_api.BaseApiService):
    """Service class for the services_projects_global_networks_peeredDnsDomains resource."""
    _NAME = 'services_projects_global_networks_peeredDnsDomains'

    def __init__(self, client):
        super(ServicenetworkingV1.ServicesProjectsGlobalNetworksPeeredDnsDomainsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a peered DNS domain which sends requests for records in given namespace originating in the service producer VPC network to the consumer VPC network to be resolved.

      Args:
        request: (ServicenetworkingServicesProjectsGlobalNetworksPeeredDnsDomainsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/services/{servicesId}/projects/{projectsId}/global/networks/{networksId}/peeredDnsDomains', http_method='POST', method_id='servicenetworking.services.projects.global.networks.peeredDnsDomains.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/peeredDnsDomains', request_field='peeredDnsDomain', request_type_name='ServicenetworkingServicesProjectsGlobalNetworksPeeredDnsDomainsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a peered DNS domain.

      Args:
        request: (ServicenetworkingServicesProjectsGlobalNetworksPeeredDnsDomainsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/services/{servicesId}/projects/{projectsId}/global/networks/{networksId}/peeredDnsDomains/{peeredDnsDomainsId}', http_method='DELETE', method_id='servicenetworking.services.projects.global.networks.peeredDnsDomains.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ServicenetworkingServicesProjectsGlobalNetworksPeeredDnsDomainsDeleteRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists peered DNS domains for a connection.

      Args:
        request: (ServicenetworkingServicesProjectsGlobalNetworksPeeredDnsDomainsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListPeeredDnsDomainsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/services/{servicesId}/projects/{projectsId}/global/networks/{networksId}/peeredDnsDomains', http_method='GET', method_id='servicenetworking.services.projects.global.networks.peeredDnsDomains.list', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/peeredDnsDomains', request_field='', request_type_name='ServicenetworkingServicesProjectsGlobalNetworksPeeredDnsDomainsListRequest', response_type_name='ListPeeredDnsDomainsResponse', supports_download=False)
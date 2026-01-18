from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.certificatemanager.v1alpha2 import certificatemanager_v1alpha2_messages as messages
class ProjectsLocationsDnsAuthorizationsService(base_api.BaseApiService):
    """Service class for the projects_locations_dnsAuthorizations resource."""
    _NAME = 'projects_locations_dnsAuthorizations'

    def __init__(self, client):
        super(CertificatemanagerV1alpha2.ProjectsLocationsDnsAuthorizationsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new DnsAuthorization in a given project and location.

      Args:
        request: (CertificatemanagerProjectsLocationsDnsAuthorizationsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/dnsAuthorizations', http_method='POST', method_id='certificatemanager.projects.locations.dnsAuthorizations.create', ordered_params=['parent'], path_params=['parent'], query_params=['dnsAuthorizationId'], relative_path='v1alpha2/{+parent}/dnsAuthorizations', request_field='dnsAuthorization', request_type_name='CertificatemanagerProjectsLocationsDnsAuthorizationsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single DnsAuthorization.

      Args:
        request: (CertificatemanagerProjectsLocationsDnsAuthorizationsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/dnsAuthorizations/{dnsAuthorizationsId}', http_method='DELETE', method_id='certificatemanager.projects.locations.dnsAuthorizations.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='CertificatemanagerProjectsLocationsDnsAuthorizationsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single DnsAuthorization.

      Args:
        request: (CertificatemanagerProjectsLocationsDnsAuthorizationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DnsAuthorization) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/dnsAuthorizations/{dnsAuthorizationsId}', http_method='GET', method_id='certificatemanager.projects.locations.dnsAuthorizations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='CertificatemanagerProjectsLocationsDnsAuthorizationsGetRequest', response_type_name='DnsAuthorization', supports_download=False)

    def List(self, request, global_params=None):
        """Lists DnsAuthorizations in a given project and location.

      Args:
        request: (CertificatemanagerProjectsLocationsDnsAuthorizationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDnsAuthorizationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/dnsAuthorizations', http_method='GET', method_id='certificatemanager.projects.locations.dnsAuthorizations.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha2/{+parent}/dnsAuthorizations', request_field='', request_type_name='CertificatemanagerProjectsLocationsDnsAuthorizationsListRequest', response_type_name='ListDnsAuthorizationsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a DnsAuthorization.

      Args:
        request: (CertificatemanagerProjectsLocationsDnsAuthorizationsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/dnsAuthorizations/{dnsAuthorizationsId}', http_method='PATCH', method_id='certificatemanager.projects.locations.dnsAuthorizations.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha2/{+name}', request_field='dnsAuthorization', request_type_name='CertificatemanagerProjectsLocationsDnsAuthorizationsPatchRequest', response_type_name='Operation', supports_download=False)
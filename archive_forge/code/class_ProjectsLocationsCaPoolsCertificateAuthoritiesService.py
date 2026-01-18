from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.privateca.v1 import privateca_v1_messages as messages
class ProjectsLocationsCaPoolsCertificateAuthoritiesService(base_api.BaseApiService):
    """Service class for the projects_locations_caPools_certificateAuthorities resource."""
    _NAME = 'projects_locations_caPools_certificateAuthorities'

    def __init__(self, client):
        super(PrivatecaV1.ProjectsLocationsCaPoolsCertificateAuthoritiesService, self).__init__(client)
        self._upload_configs = {}

    def Activate(self, request, global_params=None):
        """Activate a CertificateAuthority that is in state AWAITING_USER_ACTIVATION and is of type SUBORDINATE. After the parent Certificate Authority signs a certificate signing request from FetchCertificateAuthorityCsr, this method can complete the activation process.

      Args:
        request: (PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesActivateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Activate')
        return self._RunMethod(config, request, global_params=global_params)
    Activate.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/caPools/{caPoolsId}/certificateAuthorities/{certificateAuthoritiesId}:activate', http_method='POST', method_id='privateca.projects.locations.caPools.certificateAuthorities.activate', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:activate', request_field='activateCertificateAuthorityRequest', request_type_name='PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesActivateRequest', response_type_name='Operation', supports_download=False)

    def Create(self, request, global_params=None):
        """Create a new CertificateAuthority in a given Project and Location.

      Args:
        request: (PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/caPools/{caPoolsId}/certificateAuthorities', http_method='POST', method_id='privateca.projects.locations.caPools.certificateAuthorities.create', ordered_params=['parent'], path_params=['parent'], query_params=['certificateAuthorityId', 'requestId'], relative_path='v1/{+parent}/certificateAuthorities', request_field='certificateAuthority', request_type_name='PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Delete a CertificateAuthority.

      Args:
        request: (PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/caPools/{caPoolsId}/certificateAuthorities/{certificateAuthoritiesId}', http_method='DELETE', method_id='privateca.projects.locations.caPools.certificateAuthorities.delete', ordered_params=['name'], path_params=['name'], query_params=['ignoreActiveCertificates', 'ignoreDependentResources', 'requestId', 'skipGracePeriod'], relative_path='v1/{+name}', request_field='', request_type_name='PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Disable(self, request, global_params=None):
        """Disable a CertificateAuthority.

      Args:
        request: (PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesDisableRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Disable')
        return self._RunMethod(config, request, global_params=global_params)
    Disable.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/caPools/{caPoolsId}/certificateAuthorities/{certificateAuthoritiesId}:disable', http_method='POST', method_id='privateca.projects.locations.caPools.certificateAuthorities.disable', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:disable', request_field='disableCertificateAuthorityRequest', request_type_name='PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesDisableRequest', response_type_name='Operation', supports_download=False)

    def Enable(self, request, global_params=None):
        """Enable a CertificateAuthority.

      Args:
        request: (PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesEnableRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Enable')
        return self._RunMethod(config, request, global_params=global_params)
    Enable.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/caPools/{caPoolsId}/certificateAuthorities/{certificateAuthoritiesId}:enable', http_method='POST', method_id='privateca.projects.locations.caPools.certificateAuthorities.enable', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:enable', request_field='enableCertificateAuthorityRequest', request_type_name='PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesEnableRequest', response_type_name='Operation', supports_download=False)

    def Fetch(self, request, global_params=None):
        """Fetch a certificate signing request (CSR) from a CertificateAuthority that is in state AWAITING_USER_ACTIVATION and is of type SUBORDINATE. The CSR must then be signed by the desired parent Certificate Authority, which could be another CertificateAuthority resource, or could be an on-prem certificate authority. See also ActivateCertificateAuthority.

      Args:
        request: (PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesFetchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (FetchCertificateAuthorityCsrResponse) The response message.
      """
        config = self.GetMethodConfig('Fetch')
        return self._RunMethod(config, request, global_params=global_params)
    Fetch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/caPools/{caPoolsId}/certificateAuthorities/{certificateAuthoritiesId}:fetch', http_method='GET', method_id='privateca.projects.locations.caPools.certificateAuthorities.fetch', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:fetch', request_field='', request_type_name='PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesFetchRequest', response_type_name='FetchCertificateAuthorityCsrResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns a CertificateAuthority.

      Args:
        request: (PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CertificateAuthority) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/caPools/{caPoolsId}/certificateAuthorities/{certificateAuthoritiesId}', http_method='GET', method_id='privateca.projects.locations.caPools.certificateAuthorities.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesGetRequest', response_type_name='CertificateAuthority', supports_download=False)

    def List(self, request, global_params=None):
        """Lists CertificateAuthorities.

      Args:
        request: (PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListCertificateAuthoritiesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/caPools/{caPoolsId}/certificateAuthorities', http_method='GET', method_id='privateca.projects.locations.caPools.certificateAuthorities.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/certificateAuthorities', request_field='', request_type_name='PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesListRequest', response_type_name='ListCertificateAuthoritiesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Update a CertificateAuthority.

      Args:
        request: (PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/caPools/{caPoolsId}/certificateAuthorities/{certificateAuthoritiesId}', http_method='PATCH', method_id='privateca.projects.locations.caPools.certificateAuthorities.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='certificateAuthority', request_type_name='PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesPatchRequest', response_type_name='Operation', supports_download=False)

    def Undelete(self, request, global_params=None):
        """Undelete a CertificateAuthority that has been deleted.

      Args:
        request: (PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesUndeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Undelete')
        return self._RunMethod(config, request, global_params=global_params)
    Undelete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/caPools/{caPoolsId}/certificateAuthorities/{certificateAuthoritiesId}:undelete', http_method='POST', method_id='privateca.projects.locations.caPools.certificateAuthorities.undelete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:undelete', request_field='undeleteCertificateAuthorityRequest', request_type_name='PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesUndeleteRequest', response_type_name='Operation', supports_download=False)
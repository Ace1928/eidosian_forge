from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.privateca.v1 import privateca_v1_messages as messages
class ProjectsLocationsCaPoolsCertificatesService(base_api.BaseApiService):
    """Service class for the projects_locations_caPools_certificates resource."""
    _NAME = 'projects_locations_caPools_certificates'

    def __init__(self, client):
        super(PrivatecaV1.ProjectsLocationsCaPoolsCertificatesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create a new Certificate in a given Project, Location from a particular CaPool.

      Args:
        request: (PrivatecaProjectsLocationsCaPoolsCertificatesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Certificate) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/caPools/{caPoolsId}/certificates', http_method='POST', method_id='privateca.projects.locations.caPools.certificates.create', ordered_params=['parent'], path_params=['parent'], query_params=['certificateId', 'issuingCertificateAuthorityId', 'requestId', 'validateOnly'], relative_path='v1/{+parent}/certificates', request_field='certificate', request_type_name='PrivatecaProjectsLocationsCaPoolsCertificatesCreateRequest', response_type_name='Certificate', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns a Certificate.

      Args:
        request: (PrivatecaProjectsLocationsCaPoolsCertificatesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Certificate) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/caPools/{caPoolsId}/certificates/{certificatesId}', http_method='GET', method_id='privateca.projects.locations.caPools.certificates.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='PrivatecaProjectsLocationsCaPoolsCertificatesGetRequest', response_type_name='Certificate', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Certificates.

      Args:
        request: (PrivatecaProjectsLocationsCaPoolsCertificatesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListCertificatesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/caPools/{caPoolsId}/certificates', http_method='GET', method_id='privateca.projects.locations.caPools.certificates.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/certificates', request_field='', request_type_name='PrivatecaProjectsLocationsCaPoolsCertificatesListRequest', response_type_name='ListCertificatesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Update a Certificate. Currently, the only field you can update is the labels field.

      Args:
        request: (PrivatecaProjectsLocationsCaPoolsCertificatesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Certificate) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/caPools/{caPoolsId}/certificates/{certificatesId}', http_method='PATCH', method_id='privateca.projects.locations.caPools.certificates.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='certificate', request_type_name='PrivatecaProjectsLocationsCaPoolsCertificatesPatchRequest', response_type_name='Certificate', supports_download=False)

    def Revoke(self, request, global_params=None):
        """Revoke a Certificate.

      Args:
        request: (PrivatecaProjectsLocationsCaPoolsCertificatesRevokeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Certificate) The response message.
      """
        config = self.GetMethodConfig('Revoke')
        return self._RunMethod(config, request, global_params=global_params)
    Revoke.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/caPools/{caPoolsId}/certificates/{certificatesId}:revoke', http_method='POST', method_id='privateca.projects.locations.caPools.certificates.revoke', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:revoke', request_field='revokeCertificateRequest', request_type_name='PrivatecaProjectsLocationsCaPoolsCertificatesRevokeRequest', response_type_name='Certificate', supports_download=False)
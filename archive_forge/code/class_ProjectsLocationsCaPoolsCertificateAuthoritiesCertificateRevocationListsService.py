from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.privateca.v1 import privateca_v1_messages as messages
class ProjectsLocationsCaPoolsCertificateAuthoritiesCertificateRevocationListsService(base_api.BaseApiService):
    """Service class for the projects_locations_caPools_certificateAuthorities_certificateRevocationLists resource."""
    _NAME = 'projects_locations_caPools_certificateAuthorities_certificateRevocationLists'

    def __init__(self, client):
        super(PrivatecaV1.ProjectsLocationsCaPoolsCertificateAuthoritiesCertificateRevocationListsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Returns a CertificateRevocationList.

      Args:
        request: (PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesCertificateRevocationListsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CertificateRevocationList) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/caPools/{caPoolsId}/certificateAuthorities/{certificateAuthoritiesId}/certificateRevocationLists/{certificateRevocationListsId}', http_method='GET', method_id='privateca.projects.locations.caPools.certificateAuthorities.certificateRevocationLists.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesCertificateRevocationListsGetRequest', response_type_name='CertificateRevocationList', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesCertificateRevocationListsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/caPools/{caPoolsId}/certificateAuthorities/{certificateAuthoritiesId}/certificateRevocationLists/{certificateRevocationListsId}:getIamPolicy', http_method='GET', method_id='privateca.projects.locations.caPools.certificateAuthorities.certificateRevocationLists.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesCertificateRevocationListsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists CertificateRevocationLists.

      Args:
        request: (PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesCertificateRevocationListsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListCertificateRevocationListsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/caPools/{caPoolsId}/certificateAuthorities/{certificateAuthoritiesId}/certificateRevocationLists', http_method='GET', method_id='privateca.projects.locations.caPools.certificateAuthorities.certificateRevocationLists.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/certificateRevocationLists', request_field='', request_type_name='PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesCertificateRevocationListsListRequest', response_type_name='ListCertificateRevocationListsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Update a CertificateRevocationList.

      Args:
        request: (PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesCertificateRevocationListsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/caPools/{caPoolsId}/certificateAuthorities/{certificateAuthoritiesId}/certificateRevocationLists/{certificateRevocationListsId}', http_method='PATCH', method_id='privateca.projects.locations.caPools.certificateAuthorities.certificateRevocationLists.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='certificateRevocationList', request_type_name='PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesCertificateRevocationListsPatchRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesCertificateRevocationListsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/caPools/{caPoolsId}/certificateAuthorities/{certificateAuthoritiesId}/certificateRevocationLists/{certificateRevocationListsId}:setIamPolicy', http_method='POST', method_id='privateca.projects.locations.caPools.certificateAuthorities.certificateRevocationLists.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesCertificateRevocationListsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesCertificateRevocationListsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/caPools/{caPoolsId}/certificateAuthorities/{certificateAuthoritiesId}/certificateRevocationLists/{certificateRevocationListsId}:testIamPermissions', http_method='POST', method_id='privateca.projects.locations.caPools.certificateAuthorities.certificateRevocationLists.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='PrivatecaProjectsLocationsCaPoolsCertificateAuthoritiesCertificateRevocationListsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)
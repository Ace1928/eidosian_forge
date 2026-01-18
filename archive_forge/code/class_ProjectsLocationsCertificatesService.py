from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.certificatemanager.v1alpha2 import certificatemanager_v1alpha2_messages as messages
class ProjectsLocationsCertificatesService(base_api.BaseApiService):
    """Service class for the projects_locations_certificates resource."""
    _NAME = 'projects_locations_certificates'

    def __init__(self, client):
        super(CertificatemanagerV1alpha2.ProjectsLocationsCertificatesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new Certificate in a given project and location.

      Args:
        request: (CertificatemanagerProjectsLocationsCertificatesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/certificates', http_method='POST', method_id='certificatemanager.projects.locations.certificates.create', ordered_params=['parent'], path_params=['parent'], query_params=['certificateId'], relative_path='v1alpha2/{+parent}/certificates', request_field='certificate', request_type_name='CertificatemanagerProjectsLocationsCertificatesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single Certificate.

      Args:
        request: (CertificatemanagerProjectsLocationsCertificatesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/certificates/{certificatesId}', http_method='DELETE', method_id='certificatemanager.projects.locations.certificates.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='CertificatemanagerProjectsLocationsCertificatesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single Certificate.

      Args:
        request: (CertificatemanagerProjectsLocationsCertificatesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Certificate) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/certificates/{certificatesId}', http_method='GET', method_id='certificatemanager.projects.locations.certificates.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='CertificatemanagerProjectsLocationsCertificatesGetRequest', response_type_name='Certificate', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (CertificatemanagerProjectsLocationsCertificatesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/certificates/{certificatesId}:getIamPolicy', http_method='GET', method_id='certificatemanager.projects.locations.certificates.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1alpha2/{+resource}:getIamPolicy', request_field='', request_type_name='CertificatemanagerProjectsLocationsCertificatesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Certificates in a given project and location.

      Args:
        request: (CertificatemanagerProjectsLocationsCertificatesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListCertificatesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/certificates', http_method='GET', method_id='certificatemanager.projects.locations.certificates.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha2/{+parent}/certificates', request_field='', request_type_name='CertificatemanagerProjectsLocationsCertificatesListRequest', response_type_name='ListCertificatesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a Certificate.

      Args:
        request: (CertificatemanagerProjectsLocationsCertificatesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/certificates/{certificatesId}', http_method='PATCH', method_id='certificatemanager.projects.locations.certificates.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha2/{+name}', request_field='certificate', request_type_name='CertificatemanagerProjectsLocationsCertificatesPatchRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (CertificatemanagerProjectsLocationsCertificatesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/certificates/{certificatesId}:setIamPolicy', http_method='POST', method_id='certificatemanager.projects.locations.certificates.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha2/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='CertificatemanagerProjectsLocationsCertificatesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (CertificatemanagerProjectsLocationsCertificatesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/certificates/{certificatesId}:testIamPermissions', http_method='POST', method_id='certificatemanager.projects.locations.certificates.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha2/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='CertificatemanagerProjectsLocationsCertificatesTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)